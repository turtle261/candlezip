#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utf8_check.py — Verify a file is stored as UTF-8 and profile text-format features.

- Prints "UTF-8 Compatible" if the whole file decodes as UTF-8 (BOM allowed).
- Otherwise prints "Non UTF-8 compatible" and lists each line with invalid bytes.
- Always prints a compact "Text format profile" (ASCII, newlines, control chars, etc.)
- Exit codes: 0 = UTF-8 compatible; 1 = Non UTF-8 compatible or other error.

Usage:
  python utf8_check.py path/to/file
  python utf8_check.py -    # read from stdin
"""

from __future__ import annotations

import argparse
import io
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

# ----------------------------- UTF-8 error finder -----------------------------

def _find_utf8_errors(raw: bytes) -> list[tuple[int, int, str]]:
    """
    Scan a bytes line for invalid UTF-8 sequences.
    Returns a list of (start, end, message) where start/end are byte offsets
    within 'raw' [start:end) that caused a decode failure.
    """
    errors: list[tuple[int, int, str]] = []
    i = 0
    n = len(raw)
    while i < n:
        try:
            raw[i:].decode("utf-8", "strict")
            break
        except UnicodeDecodeError as e:
            start = i + e.start
            end = i + e.end
            if end <= start:
                end = start + 1  # avoid infinite loops
            errors.append((start, end, f"{e.reason}"))
            i = end
    return errors

# ------------------------------ Feature analysis ------------------------------

def _classify_bom(prefix: bytes) -> str | None:
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "UTF-8 BOM"
    if prefix.startswith(b"\xff\xfe\x00\x00"):
        return "UTF-32 LE BOM"
    if prefix.startswith(b"\x00\x00\xfe\xff"):
        return "UTF-32 BE BOM"
    if prefix.startswith(b"\xff\xfe"):
        return "UTF-16 LE BOM"
    if prefix.startswith(b"\xfe\xff"):
        return "UTF-16 BE BOM"
    return None

def _newline_stats(all_bytes: bytes) -> tuple[str, dict[str, int]]:
    crlf = all_bytes.count(b"\r\n")
    # Count raw CR not part of CRLF by replacing CRLF first
    cr = all_bytes.replace(b"\r\n", b"").count(b"\r")
    lf = all_bytes.replace(b"\r\n", b"").count(b"\n")
    mix = sum(x > 0 for x in (crlf, cr, lf)) > 1
    if mix:
        style = "Mixed"
    elif crlf:
        style = "CRLF"
    elif cr:
        style = "CR"
    elif lf:
        style = "LF"
    else:
        style = "None"
    return style, {"CRLF": crlf, "CR": cr, "LF": lf}

def _bytes_iter_lines(raw_stream: Iterable[bytes]) -> Iterable[tuple[int, bytes]]:
    for i, line in enumerate(raw_stream, start=1):
        yield i, line

def _is_ascii_only(chunk: bytes) -> bool:
    return all(b < 0x80 for b in chunk)

def _control_char_counts(all_bytes: bytes) -> dict[str, int]:
    # C0 0x00-0x1F, C1 0x80-0x9F; exclude TAB(0x09), LF(0x0A), CR(0x0D), FF(0x0C)
    counts: dict[str, int] = {"NUL(0x00)": 0, "C0-other": 0, "C1": 0}
    for b in all_bytes:
        if b == 0x00:
            counts["NUL(0x00)"] += 1
        elif b in (0x09, 0x0A, 0x0D, 0x0C):
            continue
        elif b < 0x20:
            counts["C0-other"] += 1
        elif 0x80 <= b <= 0x9F:
            counts["C1"] += 1
    return counts

def _indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in s.splitlines())

def analyze_text_features(file_bytes: bytes, utf8_ok: bool, had_bom: bool, decoded_text: str | None) -> list[str]:
    msgs: list[str] = []
    ascii_only = _is_ascii_only(file_bytes)
    bom = _classify_bom(file_bytes[:4])
    nl_style, nl_counts = _newline_stats(file_bytes)
    ctrl = _control_char_counts(file_bytes)

    # line length, trailing whitespace, tabs/spaces
    max_len = 0
    avg_len = 0.0
    total_len = 0
    line_count = 0
    trailing_ws_lines = 0
    tab_lines = 0
    space_indent_lines = 0

    # Iterate by splitting on universal newlines to remain bytes-agnostic for lengths
    for raw_line in file_bytes.replace(b"\r\n", b"\n").split(b"\n"):
        # Do not include final "no newline at EOF" extra count; handle naturally
        L = len(raw_line)
        max_len = max(max_len, L)
        total_len += L
        line_count += 1
        if L and (raw_line.endswith(b" ") or raw_line.endswith(b"\t")):
            trailing_ws_lines += 1
        # Leading whitespace classification
        i = 0
        while i < L and raw_line[i] in (0x20, 0x09):
            i += 1
        if i > 0:
            if 0x09 in raw_line[:i]:
                tab_lines += 1
            if 0x20 in raw_line[:i]:
                space_indent_lines += 1

    if line_count:
        avg_len = total_len / line_count

    msgs.append("Text format profile:")
    msgs.append(_indent(f"- ASCII-only: {'yes' if ascii_only else 'no'}"))
    if bom:
        msgs.append(_indent(f"- BOM: {bom}"))
    else:
        msgs.append(_indent("- BOM: none"))
    msgs.append(_indent(f"- Newlines: {nl_style} (LF={nl_counts['LF']}, CRLF={nl_counts['CRLF']}, CR={nl_counts['CR']})"))
    msgs.append(_indent(f"- Binary hints: NUL bytes={ctrl['NUL(0x00)']}"))
    msgs.append(_indent(f"- Control chars (excluding TAB/LF/CR/FF): C0={ctrl['C0-other']}, C1={ctrl['C1']}"))
    if line_count:
        msgs.append(_indent(f"- Lines: {line_count}, max length={max_len}, avg length≈{avg_len:.1f}"))
        msgs.append(_indent(f"- Indentation: lines with tabs={tab_lines}, with spaces={space_indent_lines}"))
        msgs.append(_indent(f"- Trailing whitespace: {trailing_ws_lines} line(s)"))
    else:
        msgs.append(_indent("- Lines: 0 (empty file)"))

    # If UTF-8 decodable, add Unicode-aware notes
    if utf8_ok and decoded_text is not None:
        # Normalization checks on a sample if very large to avoid heavy work
        sample = decoded_text if len(decoded_text) <= 2_000_000 else decoded_text[:2_000_000]
        nfc = unicodedata.normalize("NFC", sample)
        nfkc = unicodedata.normalize("NFKC", sample)
        nfc_ok = (sample == nfc)
        nfkc_ok = (sample == nfkc)

        # Detect noncharacters and surrogate code points in the decoded text
        noncharacters = 0
        surrogates = 0
        for ch in sample:
            cp = ord(ch)
            # noncharacters: U+FDD0..U+FDEF, and any codepoint with low 16 bits FFFE/FFFF
            if (0xFDD0 <= cp <= 0xFDEF) or ((cp & 0xFFFF) in (0xFFFE, 0xFFFF)):
                noncharacters += 1
            if 0xD800 <= cp <= 0xDFFF:
                surrogates += 1

        msgs.append(_indent("- Unicode (applies because UTF-8 decodes):"))
        msgs.append(_indent(f"• Normalization: NFC={'ok' if nfc_ok else 'needs-normalization'}, NFKC={'ok' if nfkc_ok else 'needs-normalization'}", n=4))
        msgs.append(_indent(f"• Noncharacters: {noncharacters}", n=4))
        msgs.append(_indent(f"• Lone surrogate code points (should not occur in valid text): {surrogates}", n=4))
        if not ascii_only:
            msgs.append(_indent("• Contains non-ASCII Unicode characters", n=4))
        else:
            msgs.append(_indent("• All characters are ASCII (7-bit)", n=4))

    return msgs

# --------------------------------- Checking ----------------------------------

def check_utf8_bytes(file_bytes: bytes) -> tuple[bool, list[str]]:
    """
    Byte-level UTF-8 compatibility, reporting exact invalid sequences per line.
    """
    msgs: list[str] = []
    had_error = False

    # BOM detection on the first line only when present
    has_utf8_bom = file_bytes.startswith(b"\xef\xbb\xbf")
    # Iterate by real file line boundaries to get line/byte columns
    # Use a BytesIO to preserve exact bytes
    f = io.BytesIO(file_bytes)
    for lineno, raw in _bytes_iter_lines(f):
        if lineno == 1 and has_utf8_bom and raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        errs = _find_utf8_errors(raw)
        if errs:
            had_error = True
            for (start, end, reason) in errs:
                bad = raw[start:end]
                col1 = start + 1
                col2 = end
                hex_bytes = " ".join(f"{b:02x}" for b in bad)
                msgs.append(f"Line {lineno}, bytes {col1}-{col2}: {reason}; bytes: {hex_bytes}")

    if not had_error:
        kind = "UTF-8 (with BOM)" if has_utf8_bom else "UTF-8"
        msgs.insert(0, f"✓ UTF-8 Compatible — detected {kind}.")
        return True, msgs
    else:
        msgs.insert(0, "✗ Non UTF-8 compatible — invalid UTF-8 sequences found:")
        return False, msgs

def run_checks(stream: io.BufferedReader) -> tuple[int, list[str]]:
    """
    Read full stream, run UTF-8 check + format profiling, and build messages.
    Returns (exit_code, messages).
    """
    try:
        file_bytes = stream.read()
    except Exception as e:
        return 1, [f"Error reading input: {e}"]

    utf8_ok, utf_msgs = check_utf8_bytes(file_bytes)

    decoded_text: str | None = None
    if utf8_ok:
        # Skip BOM if present for decoding view
        view = file_bytes[3:] if file_bytes.startswith(b"\xef\xbb\xbf") else file_bytes
        decoded_text = view.decode("utf-8", "strict")

    feature_msgs = analyze_text_features(file_bytes, utf8_ok=utf8_ok, had_bom=file_bytes.startswith(b"\xef\xbb\xbf"), decoded_text=decoded_text)

    # Stitch together: UTF-8 result first, then a blank line, then the profile.
    messages = utf_msgs + [""] + feature_msgs
    return (0 if utf8_ok else 1), messages

# ----------------------------------- CLI -------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Report whether a file is stored as UTF-8, list invalid sequences, and print a text-format profile."
    )
    ap.add_argument("file", nargs="?", help="Path to the file to check, or '-' for stdin")
    args = ap.parse_args()

    # Default behavior: require a file path; allow '-' for stdin.
    if not args.file:
        print("Error: please provide a file path (or '-' for stdin).", file=sys.stderr)
        return 1

    if args.file == "-":
        stream = sys.stdin.buffer
        exit_code, messages = run_checks(stream)
        for m in messages:
            print(m)
        return exit_code

    path = Path(args.file)
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        return 1

    with path.open("rb") as f:
        exit_code, messages = run_checks(f)
    for m in messages:
        print(m)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())

