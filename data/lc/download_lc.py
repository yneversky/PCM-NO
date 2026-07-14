#!/usr/bin/env python3
"""Download and verify the PCM-NO LC dataset release asset."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

DATA_FILE = "mac_cavity_id_paper_N64_T56_Re100-500_lid0.8-1.2.npz"
FILES = (DATA_FILE,)
DEFAULT_TAG = "lc-data-v1.0.0"
ENV_REPO = "PCMNO_GITHUB_REPO"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "raw"
CHECKSUM_FILE = SCRIPT_DIR / "SHA256SUMS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the LC dataset file and verify its SHA-256 hash."
    )
    parser.add_argument(
        "--repo",
        default=None,
        help=(
            "GitHub repository in OWNER/REPO form. If omitted, the script checks "
            f"${ENV_REPO} and then the current Git remote."
        ),
    )
    parser.add_argument("--tag", default=DEFAULT_TAG, help="GitHub Release tag.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Direct base URL containing the release asset; overrides --repo and --tag.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--force", action="store_true", help="Redownload an existing file.")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 verification. Not recommended for paper reproduction.",
    )
    return parser.parse_args()


def infer_repo(explicit: str | None) -> str:
    if explicit:
        return normalize_repo(explicit)
    env_value = os.environ.get(ENV_REPO)
    if env_value:
        return normalize_repo(env_value)
    try:
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=SCRIPT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        remote = ""
    if remote:
        match = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?$", remote)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
    raise RuntimeError(
        "Cannot determine the GitHub repository. Pass --repo OWNER/REPO or set "
        f"the {ENV_REPO} environment variable."
    )


def normalize_repo(value: str) -> str:
    value = value.strip().rstrip("/")
    value = re.sub(r"^https?://github\.com/", "", value)
    value = re.sub(r"\.git$", "", value)
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", value):
        raise ValueError(f"Invalid repository name: {value!r}; expected OWNER/REPO.")
    return value


def read_checksums(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Checksum file not found: {path}")
    checksums: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            raise ValueError(f"Malformed checksum line {line_no} in {path}: {raw!r}")
        filename = parts[1].lstrip("* ").strip()
        checksums[filename] = parts[0].lower()
    if DATA_FILE not in checksums:
        raise RuntimeError(
            f"{path} has no checksum entry for {DATA_FILE}. Run prepare_data_assets.py first."
        )
    return checksums


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "PCM-NO-dataset-downloader/1.0"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            received = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r  {destination.name}: {100.0 * received / total:6.2f}%", end="", flush=True)
        if total:
            print()
        partial.replace(destination)
    except (urllib.error.URLError, OSError):
        partial.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/") if args.base_url else None
    if base_url is None:
        repo = infer_repo(args.repo)
        base_url = f"https://github.com/{repo}/releases/download/{args.tag}"

    checksums = {} if args.no_verify else read_checksums(CHECKSUM_FILE)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    destination = args.output_dir / DATA_FILE
    expected = checksums.get(DATA_FILE)
    if destination.exists() and not args.force:
        if args.no_verify or sha256(destination) == expected:
            print(f"[skip] {destination}")
            print(f"LC dataset is ready in: {args.output_dir.resolve()}")
            return 0
        print(f"[redo] Existing file failed verification: {destination}")

    url = f"{base_url}/{DATA_FILE}"
    print(f"[download] {url}")
    download(url, destination)
    if expected is not None:
        actual = sha256(destination)
        if actual != expected:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {DATA_FILE}: expected {expected}, got {actual}."
            )
        print(f"[verified] {DATA_FILE}")

    print(f"LC dataset is ready in: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
