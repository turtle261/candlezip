// Copyright (C) 2025 Noah Cashin <noahc959@icloud.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

use anyhow::{Result, bail};
use candle_core::{Tensor, Device};
use std::collections::HashMap;
use std::io::{Write, Read};
use std::path::Path;

use super::lora::{LoraManager, LoraTrainingConfig, LoraTrainer, quantization::QuantizationLevel, LoraAdapter};
use crate::models::LanguageModelSession;

/// Header data for FineZip container
#[derive(Debug)]
pub struct HeaderData {
    pub bos_token_id: u32,
    pub token_count: u64,
    pub orig_len_bytes: u64,
    pub model_hash16: [u8; 16],
    pub tokenizer_hash16: [u8; 16],
    pub orig_hash16: [u8; 16],
    pub context_window: u32,
    pub vocab_size: u32,
    pub model_file_repr: String,
    pub reprime_interval: u32,
    pub use_ac: bool,
}

/// Configuration for FineZip encoding
#[derive(Debug, Clone)]
pub struct FineZipConfig {
    pub lora_config: super::lora::LoraConfig,
    pub training_config: LoraTrainingConfig,
    pub quantization: QuantizationLevel,
    pub use_ac: bool,
    pub context_window: usize,
}

/// FineZip encoder state
pub struct FineZipEncoder<M: LoraManager> {
    manager: M,
    config: FineZipConfig,
    trained_adapters: Option<HashMap<String, LoraAdapter>>,
}

impl<M: LoraManager> FineZipEncoder<M> {
    pub fn new(manager: M, config: FineZipConfig) -> Self {
        Self {
            manager,
            config,
            trained_adapters: None,
        }
    }

    /// Train LoRA adapters on input data
    pub fn train_lora(&mut self, input_tokens: &[u32]) -> Result<()> {
        let trainer = LoraTrainer::new(self.config.training_config.clone());

        // Split input into training sequences with dynamic chunking
        let chunks = self.create_training_chunks(input_tokens)?;

        // Train on chunks
        let mut all_adapters = HashMap::new();
        for chunk in chunks {
            let adapters = trainer.train(&mut self.manager, &chunk, &chunk)?;
            // Merge adapters (in practice, would need proper merging logic)
            for (name, adapter) in adapters {
                all_adapters.insert(name, adapter);
            }
        }

        self.trained_adapters = Some(all_adapters);
        Ok(())
    }

    /// Create training chunks with dynamic window sizing
    fn create_training_chunks(&self, tokens: &[u32]) -> Result<Vec<Vec<u32>>> {
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < tokens.len() {
            let end = (start + self.config.context_window).min(tokens.len());
            chunks.push(tokens[start..end].to_vec());
            start = end;
        }

        Ok(chunks)
    }

    /// Encode input tokens to ranks
    pub fn encode_to_ranks(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        if self.trained_adapters.is_none() {
            bail!("LoRA adapters must be trained before encoding");
        }

        // Apply trained adapters
        let mut manager = self.manager.clone(); // Assuming clone is available
        manager.apply_adapters(self.trained_adapters.as_ref().unwrap())?;

        let mut ranks = Vec::new();

        // Process tokens in batches
        for chunk in self.create_training_chunks(tokens)? {
            let chunk_ranks = self.encode_chunk_to_ranks(&manager, &chunk)?;
            ranks.extend(chunk_ranks);
        }

        Ok(ranks)
    }

    /// Encode a single chunk to ranks
    fn encode_chunk_to_ranks(&self, manager: &M, tokens: &[u32]) -> Result<Vec<u32>> {
        // TODO: Implement rank extraction
        // This involves:
        // 1. Forward pass through LoRA-augmented model
        // 2. Computing probability distribution
        // 3. Converting to ranks (position in sorted probabilities)

        // Placeholder: return token indices as ranks for now
        Ok((0..tokens.len()).map(|i| i as u32).collect())
    }

    /// Compress ranks using varint encoding and zstd
    pub fn compress_ranks(&self, ranks: &[u32]) -> Result<Vec<u8>> {
        use byteorder::WriteBytesExt;

        // Encode ranks as varints
        let mut varint_data = Vec::new();
        for &rank in ranks {
            write_var_u32(&mut varint_data, rank)?;
        }

        // Compress with zstd
        let mut encoder = zstd::Encoder::new(Vec::new(), 0)?;
        encoder.write_all(&varint_data)?;
        let compressed = encoder.finish()?;

        Ok(compressed)
    }

    /// Get serialized adapter data for storage
    pub fn get_adapter_data(&self) -> Result<Vec<u8>> {
        if let Some(adapters) = &self.trained_adapters {
            // Quantize if requested
            let final_adapters = if self.config.quantization != QuantizationLevel::None {
                let mut quantized = HashMap::new();
                for (name, adapter) in adapters {
                    let quantized_adapter = super::lora::quantization::quantize_adapter(
                        adapter,
                        self.config.quantization,
                    )?;
                    quantized.insert(name.clone(), quantized_adapter);
                }
                quantized
            } else {
                adapters.clone()
            };

            // Serialize to JSON
            let json = serde_json::to_vec(&final_adapters)?;
            Ok(json)
        } else {
            bail!("No trained adapters available");
        }
    }
}

/// Write a u32 as a variable-length integer
fn write_var_u32<W: Write>(writer: &mut W, mut value: u32) -> Result<()> {
    while value >= 0x80 {
        writer.write_all(&[((value as u8) & 0x7F) | 0x80])?;
        value >>= 7;
    }
    writer.write_all(&[value as u8])?;
    Ok(())
}

/// FineZip decoding utilities
pub mod decode {
    use super::*;
    use std::io::Read;

    /// FineZip decoder state
    pub struct FineZipDecoder<M: LoraManager> {
        manager: M,
        config: FineZipConfig,
    }

    impl<M: LoraManager> FineZipDecoder<M> {
        pub fn new(manager: M, config: FineZipConfig) -> Result<Self> {
            Ok(Self { manager, config })
        }

        /// Load and apply LoRA adapters
        pub fn load_adapters(&mut self, adapter_data: &[u8]) -> Result<()> {
            let adapters: HashMap<String, LoraAdapter> = serde_json::from_slice(adapter_data)?;

            // Dequantize if needed
            let final_adapters = if self.config.quantization != QuantizationLevel::None {
                let mut dequantized = HashMap::new();
                for (name, adapter) in adapters {
                    let dequantized_adapter = super::lora::quantization::dequantize_adapter(&adapter)?;
                    dequantized.insert(name, dequantized_adapter);
                }
                dequantized
            } else {
                adapters
            };

            self.manager.apply_adapters(&final_adapters)?;
            Ok(())
        }

        /// Decompress ranks from stored data
        pub fn decompress_ranks(&self, compressed_data: &[u8]) -> Result<Vec<u32>> {
            // Decompress zstd
            let mut decoder = zstd::Decoder::new(compressed_data)?;
            let mut varint_data = Vec::new();
            decoder.read_to_end(&mut varint_data)?;

            // Decode varints
            let mut ranks = Vec::new();
            let mut reader = &varint_data[..];
            while !reader.is_empty() {
                let (rank, remaining) = read_var_u32(reader)?;
                ranks.push(rank);
                reader = remaining;
            }

            Ok(ranks)
        }

        /// Decode ranks back to tokens
        pub fn decode_from_ranks(&self, ranks: &[u32]) -> Result<Vec<u32>> {
            // TODO: Implement rank-to-token decoding
            // This involves reversing the ranking process

            // Placeholder: return ranks as tokens for now
            Ok(ranks.to_vec())
        }
    }

    /// Read a u32 from variable-length encoding
    fn read_var_u32(data: &[u8]) -> Result<(u32, &[u8])> {
        let mut result = 0u32;
        let mut shift = 0;
        let mut index = 0;

        loop {
            if index >= data.len() {
                bail!("Unexpected end of varint data");
            }

            let byte = data[index];
            result |= ((byte & 0x7F) as u32) << shift;

            if (byte & 0x80) == 0 {
                return Ok((result, &data[index + 1..]));
            }

            shift += 7;
            index += 1;

            if shift > 28 {
                bail!("Varint too long");
            }
        }
    }
}