"""Stage the build context of the review job's image: ``python -m src.cloud.stage OUT_DIR``.

The image needs the code and exactly the data files the library is built from. Which files those are is found by
building the library here with ``open`` watched, so the list cannot go stale. Videos, the comment database and
everything else under ``data/`` stay out of the image.
"""

from __future__ import annotations

import builtins
import io
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def library_files() -> list[Path]:
    """Files under ``data/`` that are opened while the library (``tool.state()``) is built."""
    data, seen, real = str(ROOT / "data") + os.sep, set(), builtins.open

    def watching(file, *args, **kwargs):
        try:
            path = os.path.abspath(os.fspath(file))
            if path.startswith(data):
                seen.add(path)
        except TypeError:
            pass
        return real(file, *args, **kwargs)

    builtins.open = io.open = watching
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from src.tools.predict_retention import tool

        tool.state()
    finally:
        builtins.open = io.open = real
    return sorted(Path(p) for p in seen if os.path.isfile(p))


def main(argv: list[str] | None = None) -> int:
    out = Path((argv or sys.argv[1:])[0]).resolve()
    os.chdir(ROOT)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(ROOT / "src", out / "src", ignore=CODE_SKIP)
    shutil.copy(ROOT / "requirements.txt", out / "requirements.txt")
    shutil.copy(ROOT / "src" / "cloud" / "Dockerfile", out / "Dockerfile")
    files = library_files()
    for path in files:
        dest = out / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest)
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"staged {out}: code, requirements and {len(files)} library files ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
