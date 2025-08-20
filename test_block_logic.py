#!/usr/bin/env python3
"""
Simple validation script to check that block processing logic is sound.
This validates the mathematical correctness of the block processing approach.
"""

def simulate_token_processing(total_tokens, block_size):
    """Simulate the block processing logic"""
    remaining_tokens = list(range(1, total_tokens + 1))  # Skip BOS token
    token_pos = 0
    processed_blocks = []
    
    while token_pos < len(remaining_tokens):
        # Determine block size for this iteration (handle remainder)
        current_block_size = min(block_size, len(remaining_tokens) - token_pos)
        block_end = token_pos + current_block_size
        current_block = remaining_tokens[token_pos:block_end]
        
        processed_blocks.append(current_block)
        token_pos = block_end
    
    return processed_blocks

def test_block_processing():
    """Test various scenarios to ensure correctness"""
    test_cases = [
        (10, 3),   # 10 tokens, block size 3
        (32, 8),   # 32 tokens, block size 8
        (100, 32), # 100 tokens, block size 32
        (7, 10),   # Fewer tokens than block size
    ]
    
    for total_tokens, block_size in test_cases:
        blocks = simulate_token_processing(total_tokens, block_size)
        
        # Validate all tokens are processed exactly once
        flattened = [token for block in blocks for token in block]
        expected = list(range(1, total_tokens + 1))
        
        print(f"Total tokens: {total_tokens}, Block size: {block_size}")
        print(f"  Blocks: {len(blocks)}")
        print(f"  Block sizes: {[len(block) for block in blocks]}")
        print(f"  Tokens processed correctly: {flattened == expected}")
        print()

if __name__ == "__main__":
    test_block_processing()