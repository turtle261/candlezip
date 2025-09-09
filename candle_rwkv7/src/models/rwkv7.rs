use anyhow::Result;
use candle::{DType, Device, Module, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, linear_no_bias};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize, // C
    pub num_hidden_layers: usize,
    #[serde(alias = "head_size")]
    pub head_dim: usize, // N - head dimension size
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_a_low_rank_dim")]
    pub a_low_rank_dim: usize,
    #[serde(default = "default_decay_low_rank_dim")]
    pub decay_low_rank_dim: usize,
    #[serde(default = "default_v_low_rank_dim")]
    pub v_low_rank_dim: usize,
    #[serde(default = "default_gate_low_rank_dim")]
    pub gate_low_rank_dim: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
}

fn default_norm_eps() -> f64 { 1e-5 }
fn default_a_low_rank_dim() -> usize { 64 }
fn default_decay_low_rank_dim() -> usize { 64 }
fn default_v_low_rank_dim() -> usize { 32 }
fn default_gate_low_rank_dim() -> usize { 128 }

#[derive(Debug, Clone)]
pub struct StatePerLayer {
    pub att_x_prev: Tensor,           // (B, C)
    pub att_state: Tensor,            // (B, H, N, N) fp32
    pub ffn_x_prev: Tensor,           // (B, C)
}

#[derive(Debug, Clone)]
pub struct State {
    pub per_layer: Vec<StatePerLayer>,
    pub pos: usize,
}

impl State {
    pub fn new(batch_size: usize, cfg: &Config, vb: Option<VarBuilder>, dev: &Device) -> Result<Self> {
        let mut per_layer = Vec::with_capacity(cfg.num_hidden_layers);
        let dtype = vb.as_ref().map(|vb| vb.dtype()).unwrap_or(DType::F32);
        let h = cfg.hidden_size / cfg.head_dim;
        for _ in 0..cfg.num_hidden_layers {
            per_layer.push(StatePerLayer {
                att_x_prev: Tensor::zeros((batch_size, cfg.hidden_size), dtype, dev)?,
                att_state: Tensor::zeros((batch_size, h, cfg.head_dim, cfg.head_dim), DType::F32, dev)?,
                ffn_x_prev: Tensor::zeros((batch_size, cfg.hidden_size), dtype, dev)?,
            });
        }
        Ok(Self { per_layer, pos: 0 })
    }
}

// No generic LoRaMLP: RWKV7 uses explicit low-rank (w1,w2), (a1,a2), (v1,v2), (g1,g2)

#[derive(Debug, Clone)]
struct SelfAttention {
    // projections
    receptance: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    // group norm params
    gnorm_weight: Tensor,
    gnorm_bias: Tensor,
    // token shift mix
    x_r: Tensor,
    x_w: Tensor,
    x_k: Tensor,
    x_v: Tensor,
    x_a: Tensor,
    x_g: Tensor,
    // low-rank tensors for w,a,v,g
    w1_w: Tensor, // (Dw, C) stored
    w2_w: Tensor, // (C, Dw) stored
    w0_b: Tensor, // (C) - this is w0 in the Python code
    a1_w: Tensor, // (Da, C) stored
    a2_w: Tensor, // (C, Da) stored
    a0_b: Tensor, // (C)
    v1_w: Option<Tensor>, // (Dv, C) stored
    v2_w: Option<Tensor>, // (C, Dv) stored
    v0_b: Option<Tensor>, // (C)
    g1_w: Tensor, // (Dg, C) stored
    g2_w: Tensor, // (C, Dg) stored
    // key scalers
    k_k: Tensor,
    k_a: Tensor,
    r_k: Tensor, // (H, N)
    // meta
    layer_id: usize,
    heads: usize,
    head_size: usize,
}

impl SelfAttention {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let heads = cfg.hidden_size / cfg.head_dim;
        let vb = vb.pp("attn");
        let receptance = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("r_proj"))?;
        let key = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?;
        let value = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?;
        let output = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))?;

        let gnorm_weight = vb.get(cfg.hidden_size, "g_norm.weight")?;
        let gnorm_bias = vb.get(cfg.hidden_size, "g_norm.bias")?;

        let x_r = vb.get(cfg.hidden_size, "x_r")?.reshape((1, 1, cfg.hidden_size))?;
        let x_w = vb.get(cfg.hidden_size, "x_w")?.reshape((1, 1, cfg.hidden_size))?;
        let x_k = vb.get(cfg.hidden_size, "x_k")?.reshape((1, 1, cfg.hidden_size))?;
        let x_v = vb.get(cfg.hidden_size, "x_v")?.reshape((1, 1, cfg.hidden_size))?;
        let x_a = vb.get(cfg.hidden_size, "x_a")?.reshape((1, 1, cfg.hidden_size))?;
        let x_g = vb.get(cfg.hidden_size, "x_g")?.reshape((1, 1, cfg.hidden_size))?;

        let w1_w = vb.get((cfg.decay_low_rank_dim, cfg.hidden_size), "w_lora.lora.0.weight")?;
        let w2_w = vb.get((cfg.hidden_size, cfg.decay_low_rank_dim), "w_lora.lora.2.weight")?;
        let w0_b = vb.get(cfg.hidden_size, "w_lora.lora.2.bias")?; // This is w0, not w2_b!

        let a1_w = vb.get((cfg.a_low_rank_dim, cfg.hidden_size), "a_lora.lora.0.weight")?;
        let a2_w = vb.get((cfg.hidden_size, cfg.a_low_rank_dim), "a_lora.lora.2.weight")?;
        let a0_b = vb.get(cfg.hidden_size, "a_lora.lora.2.bias")?;

        let (v1_w, v2_w, v0_b) = if vb.contains_tensor("v_lora.lora.0.weight") {
            (
                Some(vb.get((cfg.v_low_rank_dim, cfg.hidden_size), "v_lora.lora.0.weight")?),
                Some(vb.get((cfg.hidden_size, cfg.v_low_rank_dim), "v_lora.lora.2.weight")?),
                Some(vb.get(cfg.hidden_size, "v_lora.lora.2.bias")?)
            )
        } else { (None, None, None) };

        let g1_w = vb.get((cfg.gate_low_rank_dim, cfg.hidden_size), "g_lora.lora.0.weight")?;
        let g2_w = vb.get((cfg.hidden_size, cfg.gate_low_rank_dim), "g_lora.lora.2.weight")?;

        let k_k = vb.get(cfg.hidden_size, "k_k")?.reshape((1, 1, cfg.hidden_size))?;
        let k_a = vb.get(cfg.hidden_size, "k_a")?.reshape((1, 1, cfg.hidden_size))?;
        let r_k = vb.get((heads, cfg.head_dim), "r_k")?;

        Ok(Self {
            receptance,
            key,
            value,
            output,
            gnorm_weight,
            gnorm_bias,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            w1_w,
            w2_w,
            w0_b,
            a1_w,
            a2_w,
            a0_b,
            v1_w,
            v2_w,
            v0_b,
            g1_w,
            g2_w,
            k_k,
            k_a,
            r_k,
            layer_id,
            heads,
            head_size: cfg.head_dim,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut StatePerLayer, v_first: &mut Option<Tensor>) -> Result<Tensor> {
        let (b, t, c) = xs.dims3()?;
        let h = self.heads;
        let n = self.head_size;

        // time shift: xx = shift(x) - x (following Python exactly)
        let xx = if t > 1 {
            let x_prev = xs.narrow(D::Minus2, 0, t - 1)?; // (B, T-1, C)
            // Use the previous state for the first token, not zeros
            let pad = state.att_x_prev.unsqueeze(1)?; // (B, 1, C)
            let shifted = Tensor::cat(&[pad, x_prev], D::Minus2)?; // (B,T,C)
            (shifted - xs)?
        } else {
            let x_prev = state.att_x_prev.unsqueeze(1)?; // (B,1,C)
            (x_prev - xs)?
        };

        // mixed inputs
        let xr = xs.add(&xx.broadcast_mul(&self.x_r)?)?;
        let xw = xs.add(&xx.broadcast_mul(&self.x_w)?)?;
        let xk = xs.add(&xx.broadcast_mul(&self.x_k)?)?;
        let xv = xs.add(&xx.broadcast_mul(&self.x_v)?)?;
        let xa = xs.add(&xx.broadcast_mul(&self.x_a)?)?;
        let xg = xs.add(&xx.broadcast_mul(&self.x_g)?)?;

        // projections
        let r = self.receptance.forward(&xr)?; // (B,T,C)
        // low-rank w per train_temp reference:
        // w_tensor = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
        // w_decay  = exp(-exp(w_tensor))
        let xw_flat = xw.reshape((b * t, c))?; // (B*T, C)
        let w_lora_flat = xw_flat.matmul(&self.w1_w.t()?)?.tanh()?.matmul(&self.w2_w.t()?)?; // (B*T, C)
        let w_lora = w_lora_flat.reshape((b, t, c))?;
        let w_combined = w_lora.broadcast_add(&self.w0_b.reshape((1, 1, c))?)?; // (B,T,C)
        // Compute w_tensor and decay in f32 for stability
        // softplus(z) = ln(1 + exp(z))
        let neg_w = w_combined.neg()?.to_dtype(DType::F32)?; // -x
        let exp_neg_w = neg_w.exp()?;
        let ones = Tensor::new(1f32, xs.device())?.broadcast_as(exp_neg_w.shape())?;
        let sp = exp_neg_w.add(&ones)?.log()?; // softplus(-x) = ln(1 + exp(-x))
        let minus_half = Tensor::new(-0.5f32, xs.device())?.broadcast_as(sp.shape())?;
        let w_tensor = sp.neg()?.add(&minus_half)?; // -softplus(-x) - 0.5
        let w_decay_all = w_tensor.exp()?.neg()?.exp()?; // exp(-exp(w_tensor)) -> (B,T,C)

        let k = self.key.forward(&xk)?; // (B,T,C)
        let mut v = self.value.forward(&xv)?; // (B,T,C)

        // low-rank a
        let xa_flat = xa.reshape((b * t, c))?;
        let a_flat = xa_flat.matmul(&self.a1_w.t()?)?.matmul(&self.a2_w.t()?)?;
        let a_flat = a_flat.broadcast_add(&self.a0_b.reshape((1, c))?)?;
        let a = candle_nn::ops::sigmoid(&a_flat.reshape((b, t, c))?)?;

        // low-rank g
        let xg_flat = xg.reshape((b * t, c))?;
        let g_flat = candle_nn::ops::sigmoid(&xg_flat.matmul(&self.g1_w.t()?)?)?.matmul(&self.g2_w.t()?)?;
        let g = g_flat.reshape((b, t, c))?;

        // value residual (layer > 0)
        if self.layer_id == 0 {
            *v_first = Some(v.clone());
        } else if let Some(vf) = v_first {
            if let (Some(v1w), Some(v2w), Some(v0b)) = (&self.v1_w, &self.v2_w, &self.v0_b) {
                let xv_flat = xv.reshape((b * t, c))?;
                let nu_flat = xv_flat.matmul(&v1w.t()?)?.matmul(&v2w.t()?)?.broadcast_add(&v0b.reshape((1, c))?)?;
                let nu = candle_nn::ops::sigmoid(&nu_flat.reshape((b, t, c))?)?;
                let delta = vf.sub(&v)?;
                v = v.add(&delta.broadcast_mul(&nu)?)?;
            }
        }

        // keys scaling and normalization per head (following Python exactly)
        let kk = k.broadcast_mul(&self.k_k)?; // (B,T,C)
        let kk_reshaped = kk.reshape((b, t, h, n))?; // (B,T,H,N)
        // L2 normalize along the last dimension (N) like numpy: kk /= max(||kk||, 1e-12)
        let kk_norm_sq = kk_reshaped.sqr()?.sum_keepdim(D::Minus1)?; // (B,T,H,1)
        let kk_norm = kk_norm_sq.sqrt()?; // sqrt first
        let clamp = Tensor::new(1e-12f32, xs.device())?.broadcast_as(kk_norm.shape())?;
        let kk_norm = kk_norm.maximum(&clamp)?; // max(norm, 1e-12)
        let kk_normalized = kk_reshaped.broadcast_div(&kk_norm)?; // (B,T,H,N)
        let kk = kk_normalized.reshape((b, t, c))?; // Back to (B,T,C)
        
        // Convert kk to proper shape for sequential processing
        let kk = kk.reshape((b, t, h, n))?.permute((0, 2, 1, 3))?; // (B,H,T,N)
        let ones = Tensor::ones_like(&a)?;
        let scale = ones.add(&a.sub(&ones)?.broadcast_mul(&self.k_a)?)?;
        let k = k.broadcast_mul(&scale)?; // k = k * (1 + (a-1)*k_a)

        // CRITICAL: For t=1 (single token), we do the matrix update like numpy reference
        // Following rwkv_v7_numpy.py exactly: S = S * w.mT - S @ kk * (kk*a).mT + v * k.mT
        
        if t == 1 {
            // Single token case - match train_temp exactly
            // w_decay = exp(-exp(w_tensor)) computed above
            let w_single = w_decay_all.clone(); // (B,T,C) with T=1
            
            // Reshape everything to (B,H,N,1) like reference: r,w,k,v,kk,a,r_k
            let w_hn1 = w_single.reshape((b, h, n, 1))?; // (B,H,N,1)
            let r_hn1 = r.reshape((b, h, n, 1))?; // (B,H,N,1)
            let k_hn1 = k.reshape((b, h, n, 1))?; // (B,H,N,1)
            let v_hn1 = v.reshape((b, h, n, 1))?; // (B,H,N,1)
            let kk_hn1 = kk.reshape((b, h, n, 1))?; // (B,H,N,1)
            let a_hn1 = a.reshape((b, h, n, 1))?; // (B,H,N,1)
            let r_k_hn1 = self.r_k.reshape((1, h, n, 1))?; // (1,H,N,1)
            
            // State update: S = S * w.mT - S @ kk * (kk*a).mT + v * k.mT
            let mut state_mat = state.att_state.clone(); // (B,H,N,N)
            
            // Term 1: S * w.mT (broadcast multiply)
            let w_decay = state_mat.broadcast_mul(&w_hn1.transpose(D::Minus2, D::Minus1)?)?; // (B,H,N,N) * (B,H,1,N)
            
            // Term 2: - S @ kk * (kk*a).mT 
            let kk_times_a = kk_hn1.broadcast_mul(&a_hn1)?; // (B,H,N,1)
            let term2_inner = state_mat.matmul(&kk_hn1)?; // (B,H,N,N) @ (B,H,N,1) = (B,H,N,1)
            let term2 = term2_inner.matmul(&kk_times_a.transpose(D::Minus2, D::Minus1)?)?; // (B,H,N,1) @ (B,H,1,N) = (B,H,N,N)
            
            // Term 3: + v * k.mT
            let term3 = v_hn1.matmul(&k_hn1.transpose(D::Minus2, D::Minus1)?)?; // (B,H,N,1) @ (B,H,1,N) = (B,H,N,N)
            
            // Combine: S = S * w.mT - S @ kk * (kk*a).mT + v * k.mT
            state_mat = w_decay.sub(&term2)?.add(&term3)?;
            
            // Output: y = S @ r
            let out = state_mat.matmul(&r_hn1)?.squeeze(D::Minus1)?; // (B,H,N,N) @ (B,H,N,1) = (B,H,N)
            let out = out.unsqueeze(1)?; // (B,1,H,N) for time dimension
            
            // Update state for next call
            state.att_state = state_mat;
            state.att_x_prev = xs.narrow(D::Minus2, t - 1, 1)?.squeeze(D::Minus2)?;
            
            let mut x = out; // (B,1,H,N)
            
            // group norm exactly like reference
            let x2 = x.reshape((b * 1, h * n))?;
            let x2 = group_norm(&x2, &self.gnorm_weight, &self.gnorm_bias, h, 64e-5)?;
            x = x2.reshape((b, 1, h, n))?;
            
            // add head-qk term: ((r * k * r_k).sum_over_N) * v
            let rkr = r_hn1.broadcast_mul(&k_hn1)?.broadcast_mul(&r_k_hn1)?; // (B,H,N,1)
            let alpha = rkr.sum_keepdim(D::Minus2)?; // (B,H,1,1)
            let add = alpha.broadcast_mul(&v_hn1)?; // (B,H,1,1) * (B,H,N,1) = (B,H,N,1)
            let add = add.squeeze(D::Minus1)?.unsqueeze(1)?; // (B,1,H,N)
            x = x.add(&add)?;
            
            let x = x.reshape((b, t, c))?;
            let x = x.broadcast_mul(&g)?; // gate
            let x = self.output.forward(&x)?;
            
            return Ok(x);
        }
        
        // Multi-token case - sequential processing (for parallel training)
        let mut state_mat = state.att_state.clone(); // (B,H,N,N) fp32

        // w decay for sequential update per train_temp: w = exp(-exp(w_tensor))
        let w_seq = w_decay_all.reshape((b, t, h, n))?.permute((0, 2, 1, 3))?; // (B,H,T,N)

        let r_hn = r.reshape((b, t, h, n))?.permute((0, 2, 1, 3))?; // (B,H,T,N)
        let v_hn = v.reshape((b, t, h, n))?.permute((0, 2, 1, 3))?; // (B,H,T,N)
        let k_hn = k.reshape((b, t, h, n))?.permute((0, 2, 1, 3))?; // (B,H,T,N)
        let a_hn = a.reshape((b, t, h, n))?.permute((0, 2, 1, 3))?; // (B,H,T,N)

        let mut out_buf: Option<Tensor> = None; // (B,T,H,N)
        for ti in 0..t {
            let r_t = r_hn.narrow(D::Minus2, ti, 1)?.squeeze(D::Minus2)?; // (B,H,N)
            let w_t = w_seq.narrow(D::Minus2, ti, 1)?.squeeze(D::Minus2)?; // (B,H,N)
            let k_t = k_hn.narrow(D::Minus2, ti, 1)?.squeeze(D::Minus2)?; // (B,H,N)
            let v_t = v_hn.narrow(D::Minus2, ti, 1)?.squeeze(D::Minus2)?; // (B,H,N)
            let kk_t = kk.narrow(D::Minus2, ti, 1)?.squeeze(D::Minus2)?; // (B,H,N)
            let a_t = a_hn.narrow(D::Minus2, ti, 1)?.squeeze(D::Minus2)?; // (B,H,N)

            // vk = v @ k^T -> (B,H,N,1) @ (B,H,1,N) = (B,H,N,N)
            let vk = v_t.unsqueeze(D::Minus1)? // (B,H,N,1)
                .matmul(&k_t.unsqueeze(D::Minus2)?)?; // (B,H,1,N) -> (B,H,N,N)

            // ab = (-kk) @ ((kk*a))^T
            let ab = kk_t.neg()?.unsqueeze(D::Minus1)? // (B,H,N,1)
                .matmul(&kk_t.broadcast_mul(&a_t)?.unsqueeze(D::Minus2)?)?; // (B,H,1,N) => (B,H,N,N)

            // state update: state = state * w + state @ ab + vk
            let state_decay = state_mat.broadcast_mul(&w_t.unsqueeze(D::Minus2)?)?; // (B,H,1,N) broadcast over rows
            let state_ab = state_mat.matmul(&ab)?; // (B,H,N,N)
            state_mat = state_decay.add(&state_ab)?.add(&vk)?;

            // output: (state @ r)
            let out_t = state_mat.to_dtype(xs.dtype())?
                .matmul(&r_t.unsqueeze(D::Minus1)?)? // (B,H,N,1)
                .squeeze(D::Minus1)?; // (B,H,N)
            let out_t = out_t.unsqueeze(2)?; // (B,H,1,N) time dim at axis 2
            out_buf = Some(match out_buf { None => out_t, Some(prev) => Tensor::cat(&[prev, out_t], 2)? });
        }

        let mut x = out_buf.unwrap().permute((0, 2, 1, 3))?; // (B,T,H,N)
        // group norm over channels (H groups), eps = 64e-5 per reference
        let x2 = x.reshape((b * t, h * n))?;
        let x2 = group_norm(&x2, &self.gnorm_weight, &self.gnorm_bias, h, 64e-5)?;
        x = x2.reshape((b, t, h, n))?;

        // add head-qk term: ((r * k * r_k).sum_over_N) * v
        let rkr = r_hn.broadcast_mul(&k_hn)?.broadcast_mul(&self.r_k.reshape((1, h, 1, n))?)?; // (B,H,T,N)
        let alpha = rkr.sum_keepdim(D::Minus1)?; // (B,H,T,1)
        let add = alpha.broadcast_mul(&v_hn)?; // (B,H,T,N)
        let add = add.permute((0, 2, 1, 3))?; // (B,T,H,N)
        x = x.add(&add)?;

        let x = x.reshape((b, t, c))?;
        let x = x.broadcast_mul(&g)?; // gate
        let x = self.output.forward(&x)?;

        // update state for next call
        state.att_state = state_mat;
        state.att_x_prev = xs.narrow(D::Minus2, t - 1, 1)?.squeeze(D::Minus2)?;

        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    x_k: Tensor,
    key: Linear,
    value: Linear,
}

impl FeedForward {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let x_k = vb.get(cfg.hidden_size, "x_k")?.reshape((1, 1, cfg.hidden_size))?;
        let hidden = cfg.intermediate_size.unwrap_or(4 * cfg.hidden_size);
        let key = linear_no_bias(cfg.hidden_size, hidden, vb.pp("key"))?;
        let value = linear_no_bias(hidden, cfg.hidden_size, vb.pp("value"))?;
        Ok(Self { x_k, key, value })
    }

    fn forward(&self, xs: &Tensor, state: &mut StatePerLayer) -> Result<Tensor> {
        let (_b, t, _c) = xs.dims3()?;
        // time shift: xx = shift(x) - x (using previous state like attention)
        let xx = if t > 1 {
            let x_prev = xs.narrow(D::Minus2, 0, t - 1)?; // (B, T-1, C)
            // Use the previous state for the first token, not zeros
            let pad = state.ffn_x_prev.unsqueeze(1)?; // (B, 1, C)
            let shifted = Tensor::cat(&[pad, x_prev], D::Minus2)?; // (B,T,C)
            (shifted - xs)?
        } else {
            let x_prev = state.ffn_x_prev.unsqueeze(1)?; // (B,1,C)
            (x_prev - xs)?
        };
        let xk = xs.add(&xx.broadcast_mul(&self.x_k)?)?;
        let k = self.key.forward(&xk)?.relu()?.sqr()?;
        let v = self.value.forward(&k)?;
        state.ffn_x_prev = xs.narrow(D::Minus2, t - 1, 1)?.squeeze(D::Minus2)?;
        Ok(v)
    }
}

#[derive(Debug, Clone)]
struct Block {
    pre_ln: Option<LayerNorm>,
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention: SelfAttention,
    feed_forward: FeedForward,
}

impl Block {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let pre_ln = if layer_id == 0 {
            Some(LayerNorm::new(
                vb.get(cfg.hidden_size, "pre_norm.weight")?,
                vb.get(cfg.hidden_size, "pre_norm.bias")?,
                cfg.norm_eps,
            ))
        } else {
            None
        };
        let ln1 = LayerNorm::new(
            vb.get(cfg.hidden_size, "attn_norm.weight")?,
            vb.get(cfg.hidden_size, "attn_norm.bias")?,
            cfg.norm_eps,
        );
        let ln2 = LayerNorm::new(
            vb.get(cfg.hidden_size, "ffn_norm.weight")?,
            vb.get(cfg.hidden_size, "ffn_norm.bias")?,
            cfg.norm_eps,
        );
        let attention = SelfAttention::new(layer_id, cfg, vb.clone())?;
        let feed_forward = FeedForward::new(cfg, vb.pp("ffn"))?;
        Ok(Self { pre_ln, ln1, ln2, attention, feed_forward })
    }

    fn forward(&self, xs: &Tensor, state: &mut StatePerLayer, v_first: &mut Option<Tensor>) -> Result<Tensor> {
        let mut x = xs.clone();
        if let Some(pre_ln) = &self.pre_ln {
            x = pre_ln.forward(&x)?;
        }
        let att = self.attention.forward(&self.ln1.forward(&x)?, state, v_first)?;
        x = x.add(&att)?;
        let ffn = self.feed_forward.forward(&self.ln2.forward(&x)?, state)?;
        x = x.add(&ffn)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embedding,
    blocks: Vec<Block>,
    ln_out: LayerNorm,
    head_w: Tensor,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embeddings = Embedding::new(
            vb.get((cfg.vocab_size, cfg.hidden_size), "model.embeddings.weight")?,
            cfg.hidden_size,
        );
        let ln_out = LayerNorm::new(
            vb.get(cfg.hidden_size, "model.norm.weight")?,
            vb.get(cfg.hidden_size, "model.norm.bias")?,
            cfg.norm_eps,
        );
        let head_w = vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb = vb.pp(&format!("model.layers.{i}"));
            blocks.push(Block::new(i, cfg, vb)?);
        }
        Ok(Self { embeddings, blocks, ln_out, head_w })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (_b, _t) = xs.dims2()?;
        let mut x = self.embeddings.forward(xs)?;
        let mut v_first = None;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, &mut state.per_layer[i], &mut v_first)?;
        }
        x = self.ln_out.forward(&x)?;
        // Safe head matmul: flatten time then restore
        let (b, t, c) = x.dims3()?;
        let x2d = x.reshape((b * t, c))?;
        let out2d = x2d.matmul(&self.head_w.t()?)?; // (B*T, V)
        let out = out2d.reshape((b, t, self.head_w.dims2()?.0))?;
        state.pos += t;
        Ok(out)
    }
}



fn group_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
    // Match numpy reference exactly: ((x - mean) / sqrt(var + eps)) * w + b
    // where mean/var are along channel-per-group axis using population variance
    let (n, c) = x.dims2()?;
    let channels_per_group = c / groups;
    let x = x.reshape((n, groups, channels_per_group))?;
    let mean = x.mean_keepdim(2)?; // (n, groups, 1)
    let centered = x.broadcast_sub(&mean)?; // (n, groups, channels_per_group)
    let var = centered.sqr()?.mean_keepdim(2)?; // population variance (mean of squares)
    let eps_t = Tensor::new(eps as f32, x.device())?.broadcast_as(var.shape())?;
    let denom = var.add(&eps_t)?.sqrt()?;
    let norm = centered.broadcast_div(&denom)?; // (n, groups, channels_per_group)
    Ok(norm.reshape((n, c))?.broadcast_mul(weight)?.broadcast_add(bias)?)
}

type Bytes = Vec<u8>;

pub struct Tokenizer {
    table: Vec<Vec<Vec<Bytes>>>,
    good: Vec<HashSet<u8>>,
    idx2token: HashMap<u32, Vec<u8>>,
    token2idx: HashMap<Vec<u8>, u32>,
}

impl Tokenizer {
    pub fn new<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let file = std::fs::File::open(p)?;
        let token2idx: HashMap<String, u32> = serde_json::from_reader(file).map_err(candle::Error::wrap)?;
        let token2idx = token2idx.into_iter().map(|(key, value)| (key.into_bytes(), value)).collect::<HashMap<_, _>>();
        let idx2token = token2idx.iter().map(|(key, value)| (*value, key.to_vec())).collect::<HashMap<_, _>>();
        let max_idx = token2idx.values().copied().max().unwrap_or(0);
        let mut table = vec![vec![vec![]; 256]; 256];
        let mut good = vec![HashSet::new(); 256];
        for idx in (0..(1 + max_idx)).rev() {
            let s = match idx2token.get(&idx) { None => continue, Some(s) => s };
            if s.len() >= 2 {
                let (s0, s1) = (s[0], s[1]);
                table[s0 as usize][s1 as usize].push(s.to_vec());
                good[s0 as usize].insert(s1);
            }
        }
        Ok(Self { table, good, idx2token, token2idx })
    }

    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut v = Vec::new();
        for token_id in tokens.iter() { if let Some(token) = self.idx2token.get(token_id) { v.extend_from_slice(token.as_slice()) } }
        v
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let bytes = self.decode_bytes(tokens);
        String::from_utf8(bytes).map_err(|e| anyhow::anyhow!(e))
    }

    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let mut s = vec![bytes[i]];
            if i + 1 < bytes.len() && self.good[bytes[i] as usize].contains(&bytes[i + 1]) {
                let table = &self.table[bytes[i] as usize][bytes[i + 1] as usize];
                for table_elem in table.iter() {
                    if bytes[i..].starts_with(table_elem) { s = table_elem.to_vec(); break; }
                }
            }
            i += s.len();
            let token = match self.token2idx.get(&s) { 
                None => {
                    // For lossless compression, we cannot use fallback tokens as they lose information
                    // The caller should handle this by ensuring the vocabulary covers all necessary byte sequences
                    return Err(anyhow::anyhow!("Tokenizer vocabulary missing byte sequence: {:?} (as UTF-8: '{}'). This would cause lossless compression failure.", s, String::from_utf8_lossy(&s)));
                }, 
                Some(token) => *token 
            };
            tokens.push(token)
        }
        Ok(tokens)
    }

    pub fn encode(&self, str: &str) -> Result<Vec<u32>> { self.encode_bytes(str.as_bytes()) }
}


