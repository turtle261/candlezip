#!/bin/bash

# Simple performance comparison for block-based processing
# This script tests different block sizes to find optimal performance

if [ ! -f "test_input.txt" ]; then
    echo "Error: test_input.txt not found. Please create a test file."
    exit 1
fi

if [ ! -f "target/release/gptzip" ]; then
    echo "Building release version..."
    cargo build --release
fi

echo "Performance comparison for different block sizes:"
echo "================================================="

# Test with different block sizes
for block_size in 1 8 16 32 64; do
    echo "Testing with block size: $block_size"
    
    start_time=$(date +%s.%N)
    
    # Run with timeout to prevent hanging
    timeout 60 ./target/release/gptzip --cpu --context 128 --reprime-interval 64 --block-size $block_size compress test_input.txt test_output_${block_size}.gptz 2>/dev/null
    
    if [ $? -eq 0 ]; then
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l)
        
        if [ -f "test_output_${block_size}.gptz" ]; then
            input_size=$(wc -c < test_input.txt)
            output_size=$(wc -c < "test_output_${block_size}.gptz")
            ratio=$(echo "scale=3; $output_size * 8.0 / $input_size" | bc -l)
            
            echo "  Duration: ${duration}s, Output: ${output_size} bytes, Ratio: ${ratio} bpb"
            rm -f "test_output_${block_size}.gptz"
        else
            echo "  Failed to create output file"
        fi
    else
        echo "  Test timed out or failed"
    fi
    
    echo ""
done

echo "Note: These tests require model download and may not work in environments without internet access."