#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python src/generate_sample_data.py
python src/run_pipeline.py "$@"
