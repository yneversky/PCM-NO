# Merge this code package into the existing PCM-NO repository

This package intentionally does not contain the existing `data/` directory.

From the repository root, extract the archive to a temporary directory and copy its contents over the repository:

```bash
unzip pcmno_code_release.zip -d /tmp/pcmno_code_release
cp -a /tmp/pcmno_code_release/pcmno_code_release/. .
```

This operation replaces the root `README.md` and adds the source package, configurations, scripts, tests, workflows, and reproduction files. It does not remove or replace `data/pns/` or `data/lc/`.

Then run:

```bash
pip install -e ".[dev]"
pytest
git status
```

Review the changes before committing:

```bash
git add README.md pyproject.toml requirements.txt .gitignore \
  configs src scripts tests reproduction notebooks .github \
  INSTALL_IN_EXISTING_REPO.md prepare_data_assets.py

git commit -m "Release PCM-NO training and evaluation code"
git push origin main
```
