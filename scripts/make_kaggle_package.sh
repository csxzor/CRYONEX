#!/usr/bin/env bash
# Build dist/kcwm-package.zip: code + configs + processed training data (~90 MB), no raw data.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist/kcwm-package && mkdir -p dist/kcwm-package/data/processed
cp -r kcwm configs third_party scripts kaggle pyproject.toml README.md dist/kcwm-package/
cp -r data/processed/windows data/processed/campaigns data/processed/manifest.json dist/kcwm-package/data/processed/
find dist/kcwm-package -name "__pycache__" -prune -exec rm -rf {} +
(cd dist && rm -f kcwm-package.zip && zip -qr kcwm-package.zip kcwm-package)
du -sh dist/kcwm-package.zip
