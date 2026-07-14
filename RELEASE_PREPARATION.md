# Preparing the PCM-NO dataset release files

This package contains all static repository files requested for `data/pns/` and
`data/lc/`. The two sample binaries and final checksum values depend on the
actual Drive datasets and therefore must be generated from those files.

From the repository root in Colab:

```bash
python prepare_data_assets.py
```

This command validates the expected paper-scale shapes, writes:

```text
data/pns/sample/pns_sample.pt
data/lc/sample/lc_sample.npz
data/pns/SHA256SUMS
data/lc/SHA256SUMS
```

and leaves the full release assets in their existing Drive locations.

To overwrite previously generated sample files:

```bash
python prepare_data_assets.py --overwrite
```

After generation, review the files, commit the repository metadata and samples,
and upload only the full datasets plus the corresponding `SHA256SUMS` files to
the two GitHub Releases.
