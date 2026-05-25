#!/usr/bin/env bash
# Build code reference docs with pdoc.
#
# Output goes to ./docs-site/ — pick that up in the GH Pages workflow and
# mount it at the /reference subpath of the deployed site.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-docs-site}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

uv run --extra docs pdoc \
  --docformat google \
  --no-show-source \
  --output-directory "$OUT_DIR" \
  --logo "https://raw.githubusercontent.com/your-handle/regime-detection/main/web/favicon.svg" \
  src/regime

echo
echo "Docs built at: $OUT_DIR/"
echo "Preview locally:  python3 -m http.server --directory $OUT_DIR 8088"
