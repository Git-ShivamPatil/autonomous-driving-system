"""Rewrite a wheel without members whose names exceed a length limit.

The MetaDrive 0.4.3 wheel ships example TensorBoard logs whose paths exceed Windows' 260-character
limit once installed under a typical venv. This drops those members (and their RECORD rows) so the
wheel installs without enabling long paths system-wide:

    pip download metadrive-simulator==0.4.3 --no-deps -d wheels
    python tools/strip_wheel_long_paths.py wheels/metadrive_simulator-0.4.3-py3-none-any.whl \
        fixed/metadrive_simulator-0.4.3-py3-none-any.whl 150
    pip install fixed/metadrive_simulator-0.4.3-py3-none-any.whl
"""

import csv
import io
import sys
import zipfile


def main(src: str, dst: str, limit: int) -> None:
    with zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        drop = {n for n in names if len(n) > limit}
        print(f"{len(names)} members, dropping {len(drop)}")
        for d in sorted({n.rsplit("/", 1)[0] for n in drop}):
            print("  ", d)
        record = next(n for n in names if n.endswith(".dist-info/RECORD"))
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename in drop:
                    continue
                data = zin.read(item.filename)
                if item.filename == record:
                    rows = [r for r in csv.reader(io.StringIO(data.decode())) if r and r[0] not in drop]
                    buf = io.StringIO()
                    csv.writer(buf, lineterminator="\n").writerows(rows)
                    data = buf.getvalue().encode()
                zout.writestr(item, data)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
