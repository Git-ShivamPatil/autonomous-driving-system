"""Read individual members of a zip archive served over HTTP, using Range requests.

BDD100K's archives are single multi-GB zips; this fetches only the central directory and the members
asked for. Used to build a small development subset without downloading 6 GB:

    python tools/remote_zip.py --out data/bdd_dev --per-split 48

The subset mirrors the layout of the extracted archives (100k/<split>/<name>.jpg, labels JSON and
drivable id maps), so ads.perception.prepare runs on it unchanged.
"""

from __future__ import annotations

import argparse
import struct
import time
import urllib.request
import zlib
from pathlib import Path

BASE = "http://dl.yf.io/bdd100k/"
IMAGES = BASE + "bdd100k_images_100k.zip"
LABELS = BASE + "bdd100k_labels.zip"
DRIVABLE = BASE + "bdd100k_drivable_maps.zip"


def _get(url: str, start: int | None = None, end: int | None = None, retries: int = 4) -> bytes:
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _size(url: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        return int(r.headers["Content-Length"])


class RemoteZip:
    def __init__(self, url: str) -> None:
        self.url = url
        n = _size(url)
        tail = _get(url, max(0, n - 128 * 1024), n - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise ValueError(f"no end-of-central-directory record in {url}")
        _, _, _, _, _, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i : i + 22])
        j = tail.rfind(b"PK\x06\x07")  # zip64 locator
        if j >= 0:
            (z64_off,) = struct.unpack("<Q", tail[j + 8 : j + 16])
            z = _get(url, z64_off, z64_off + 55)
            cd_size, cd_off = struct.unpack("<QQ", z[40:56])
        cd = _get(url, cd_off, cd_off + cd_size - 1)
        self.members: dict[str, tuple[int, int, int]] = {}  # name -> (local header offset, compressed size, method)
        p = 0
        while cd[p : p + 4] == b"PK\x01\x02":
            hdr = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p : p + 46])
            method, csize, usize, fnl, exl, cml, lho = hdr[4], hdr[8], hdr[9], hdr[10], hdr[11], hdr[12], hdr[16]
            name = cd[p + 46 : p + 46 + fnl].decode("utf-8", "replace")
            extra = cd[p + 46 + fnl : p + 46 + fnl + exl]
            q = 0
            while q + 4 <= len(extra):
                hid, hl = struct.unpack("<HH", extra[q : q + 4])
                if hid == 1:  # zip64 extended information
                    vals, k = extra[q + 4 : q + 4 + hl], 0
                    if usize == 0xFFFFFFFF:
                        k += 8
                    if csize == 0xFFFFFFFF:
                        (csize,) = struct.unpack("<Q", vals[k : k + 8])
                        k += 8
                    if lho == 0xFFFFFFFF:
                        (lho,) = struct.unpack("<Q", vals[k : k + 8])
                q += 4 + hl
            self.members[name] = (lho, csize, method)
            p += 46 + fnl + exl + cml

    def read(self, name: str) -> bytes:
        lho, csize, method = self.members[name]
        head = _get(self.url, lho, lho + 29)
        fnl, exl = struct.unpack("<HH", head[26:30])
        start = lho + 30 + fnl + exl
        raw = _get(self.url, start, start + csize - 1) if csize else b""
        if method == 0:
            return raw
        if method == 8:
            return zlib.decompress(raw, -15)
        raise ValueError(f"unsupported compression method {method} for {name}")


def fetch_dev_subset(out: Path, per_split: int) -> None:
    labels, images, drivable = RemoteZip(LABELS), RemoteZip(IMAGES), RemoteZip(DRIVABLE)
    for split in ("train", "val"):
        names = sorted(n for n in labels.members if n.startswith(f"100k/{split}/") and n.endswith(".json"))
        for label_name in names[:per_split]:
            stem = Path(label_name).stem
            wanted = {
                label_name: out / "labels" / label_name,
                f"100k/{split}/{stem}.jpg": out / "images" / f"100k/{split}/{stem}.jpg",
                f"labels/{split}/{stem}_drivable_id.png": out / "drivable" / f"labels/{split}/{stem}_drivable_id.png",
            }
            for member, dest in wanted.items():
                archive = labels if member.endswith(".json") else images if member.endswith(".jpg") else drivable
                if dest.exists() or member not in archive.members:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(member))
        print(f"{split}: {min(per_split, len(names))} images", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-split", type=int, default=48)
    args = p.parse_args()
    fetch_dev_subset(args.out, args.per_split)


if __name__ == "__main__":
    main()
