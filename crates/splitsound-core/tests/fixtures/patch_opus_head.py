#!/usr/bin/env python3
"""Rewrites fields of the OpusHead packet of an Ogg Opus file in place and
fixes the page checksum. Used to make fixtures ffmpeg cannot produce by
itself (a 44.1 kHz "original sample rate", a non-default pre-skip, a
non-zero output gain).

    python3 patch_opus_head.py <in.opus> <out.opus> [--rate HZ] [--pre-skip N] [--gain Q8]
"""
import struct
import sys


def ogg_crc(data):
    crc = 0
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def main():
    src, dst = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    opts = {args[i]: int(args[i + 1]) for i in range(0, len(args), 2)}
    data = bytearray(open(src, "rb").read())
    assert data[0:4] == b"OggS", "not an Ogg file"
    nsegs = data[26]
    body = 27 + nsegs
    page_len = body + sum(data[27:27 + nsegs])
    assert data[body:body + 8] == b"OpusHead", "first page is not OpusHead"
    if "--pre-skip" in opts:
        struct.pack_into("<H", data, body + 10, opts["--pre-skip"])
    if "--rate" in opts:
        struct.pack_into("<I", data, body + 12, opts["--rate"])
    if "--gain" in opts:
        struct.pack_into("<h", data, body + 16, opts["--gain"])
    data[22:26] = b"\0\0\0\0"
    struct.pack_into("<I", data, 22, ogg_crc(bytes(data[0:page_len])))
    open(dst, "wb").write(bytes(data))


if __name__ == "__main__":
    main()
