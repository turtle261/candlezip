#!/usr/bin/env python3
"""
Extract non-overlapping 128KB chunks from a file.
"""
import argparse
import os
from pathlib import Path

CHUNK_SIZE = 128 * 1024  # 128KB in bytes


def parse_size(size_str):
    """Convert size string (e.g., '3MB', '500KB') to bytes."""
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        return int(size_str)  # Assume bytes if no unit


def extract_chunks(input_path, total_bytes, output_dir):
    """Extract non-overlapping 128KB chunks from input file."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Validate input file
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    file_size = input_path.stat().st_size
    num_chunks = total_bytes // CHUNK_SIZE
    
    # Validate we have enough data
    if total_bytes > file_size:
        raise ValueError(f"Requested {total_bytes} bytes but file only has {file_size} bytes")
    
    if num_chunks * CHUNK_SIZE > file_size:
        raise ValueError(f"Need {num_chunks * CHUNK_SIZE} bytes but file only has {file_size} bytes")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract chunks
    base_name = input_path.stem  # Filename without extension
    
    with open(input_path, 'rb') as infile:
        for i in range(num_chunks):
            chunk = infile.read(CHUNK_SIZE)
            
            if len(chunk) != CHUNK_SIZE:
                raise RuntimeError(f"Failed to read full chunk {i}: got {len(chunk)} bytes")
            
            output_path = output_dir / f"{base_name}_128kb_{i}"
            
            with open(output_path, 'wb') as outfile:
                outfile.write(chunk)
    
    print(f"✓ Extracted {num_chunks} chunks ({total_bytes:,} bytes total)")
    print(f"✓ Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract non-overlapping 128KB chunks from a file'
    )
    parser.add_argument('--in', dest='input_file', required=True,
                        help='Input file path')
    parser.add_argument('--total', required=True,
                        help='Total size to extract (e.g., 3MB, 500KB)')
    parser.add_argument('--output', required=True,
                        help='Output directory path')
    
    args = parser.parse_args()
    
    try:
        total_bytes = parse_size(args.total)
        extract_chunks(args.input_file, total_bytes, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
