#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q
python -m src.train --config config.yaml
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
files = ['README.md', 'config.yaml', 'requirements.txt', 'run_all.sh',
         'tests/test_pipeline.py', 'logs/experiment.log']
files += [str(p) for p in sorted(Path('src').glob('*.py'))]
files += ['results/' + name for name in (
    'main_metrics.csv', 'main_metrics_seeds.csv', 'edge_metrics.csv',
    'edge_metrics_seeds.csv', 'error_cases.csv', 'train_history.csv', 'demo_out.txt',
    'fig1_data_dist.png', 'fig2_model_comp.png', 'fig3_conf_matrix.png',
    'fig4_edge_acc.png', 'fig5_attn_viz.png')]
for name in files:
    if not Path(name).is_file():
        raise FileNotFoundError(name)
with ZipFile('MFE310TC_Submission_Package.zip', 'w', ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(name)
print('Saved MFE310TC_Submission_Package.zip')
PY
