#!/usr/bin/env python3
"""Render the book's generated P5 images to PNG using only the Python standard library.

Run from any directory: python3 /data2/cuda-guide/scripts/render_images.py
The PGM values are not changed; display pixels use nearest-neighbor 4x scaling.
"""
from pathlib import Path
import struct
import zlib


def read_pgm(path):
    magic, size, maximum, pixels = path.read_bytes().split(b"\n", 3)
    width, height = map(int, size.split())
    if magic != b"P5" or maximum != b"255" or len(pixels) != width * height:
        raise ValueError(f"Unsupported PGM: {path}")
    return width, height, pixels


def png(path, width, height, pixels):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    rows = b"".join(b"\0" + pixels[y*width:(y+1)*width] for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def scaled(width, height, pixels):
    rows = []
    for y in range(height):
        row = b"".join(bytes([p])*4 for p in pixels[y*width:(y+1)*width])
        rows.extend([row]*4)
    return width*4, height*4, b"".join(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    count = 0
    for path in sorted(root.glob("chapters/*/results/images/*.pgm")):
        png(path.with_suffix(".png"), *scaled(*read_pgm(path)))
        count += 1
    folder = root / "chapters/ch05-image-filtering/results/images"
    tiles = [scaled(*read_pgm(folder / (name + ".pgm")))
             for name in ("clean", "noisy", "box", "median")]
    w, h, _ = tiles[0]
    gap = 8
    width, height = 2*w+gap, 2*h+gap
    pixels = bytearray([255]) * (width*height)
    for index, (tw, th, data) in enumerate(tiles):
        if (tw, th) != (w, h):
            raise ValueError("Contact sheet shape mismatch")
        ox, oy = (index % 2)*(w+gap), (index // 2)*(h+gap)
        for y in range(h):
            start = (oy+y)*width+ox
            pixels[start:start+w] = data[y*w:(y+1)*w]
    png(folder / "comparison.png", width, height, bytes(pixels))
    print(f"Rendered {count} PNG images and one comparison sheet")


if __name__ == "__main__":
    main()
