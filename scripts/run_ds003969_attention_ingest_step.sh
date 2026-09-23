#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/box/.local/bin:/usr/bin:$PATH"
ROOT=/workspace/muse-eeg-heads
cd "$ROOT"
source .venv/bin/activate

LOG="$ROOT/exports/ds003969_attention_ingest_run.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

python scripts/ds003969_attention_ingest.py

# docs
python - <<'PY'
from pathlib import Path
import json
from datetime import datetime, timezone
root = Path('/workspace/muse-eeg-heads')
summary = json.loads((root/'exports/ds003969_attention_ingest_summary.json').read_text())
rows = []
for s in summary['subjects']:
    c = s['counts']
    rows.append(f"| {s['tag']} | {s['group']} | {c.get('concentration',0)} | {c.get('mind_wandering',0)} | {s['n_windows_total']} |")
doc = f"""# ds003969 attention ingest (Head A)

**Step:** `ds003969_attention_ingest`  
**Completed (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}  
**Dataset:** OpenNeuro `ds003969` v1.0.0 — SPDX **CC0-1.0** — DOI `doi:10.18112/openneuro.ds003969.v1.0.0`

## What landed

Cached **3 subjects × 2 blocks** (`med1breath` + `think1`): `sub-001` / `sub-002` (htr) + `sub-025` (ctr).

| Tag | Group | concentration | mind_wandering | total |
|-----|-------|--------------:|---------------:|------:|
{chr(10).join(rows)}
| **total** | | **{summary['totals'].get('concentration',0)}** | **{summary['totals'].get('mind_wandering',0)}** | **{summary['n_windows_total']}** |

Windows: **2 s / 1.0 s hop**, edge trim **30 s**, cap **400**/block, shape `(N, 4, 512)` @ 256 Hz (resampled from 1024 Hz).

## Label rule

- Block-level protocol proxy: **`med*` → `concentration`**, **`think*` → `mind_wandering`**.
- Weaker than ds001787 probe ratings — fine for Muse-channel bootstrap with missing-class mask.
- No event-code remapping (tasks are separate BIDS runs).

## Channel proxy

**AF7 / AF8** native; **TP9/TP10 ← TP7/TP8** (dataset has no TP9/TP10). Export order Muse names.

## Artifacts

- Script: `scripts/ds003969_attention_ingest.py`
- Exports: `exports/windows_ds003969/`
- Summary: `exports/ds003969_attention_ingest_summary.json`
- Provenance: `kaggle_datasets/muse-eeg-heads-cache/data/ds003969/provenance_attention_ingest.json`
- Private windows copies under `kaggle_datasets/muse-eeg-heads-windows/ds003969_*`

## Out of scope this step

- No Head A retraining (next: attention-only smoke or joint 4-way with Muse-channel subjects).
- No remaining med2/think2 blocks or other subjects.
- Head B / Head C training untouched.
"""
(root/'docs/ds003969_attention_ingest.md').write_text(doc)
print('wrote docs')
PY

# private Kaggle version bumps (never -u)
sleep 5
kaggle datasets version -p "$ROOT/kaggle_datasets/muse-eeg-heads-windows" -m "ds003969 attention windows sub001/002/025 med1breath+think1" -r zip 2>&1 || echo "WINDOWS_VERSION_FAIL=$?"
sleep 8
kaggle datasets version -p "$ROOT/kaggle_datasets/muse-eeg-heads-cache" -m "cache ds003969 raw subset + provenance" -r zip 2>&1 || echo "CACHE_VERSION_FAIL=$?"

# update overnight_state
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path('/workspace/muse-eeg-heads/overnight_state.json')
st = json.loads(p.read_text())
summary = json.loads(Path('/workspace/muse-eeg-heads/exports/ds003969_attention_ingest_summary.json').read_text())
step = 'ds003969_attention_ingest'
if step not in st.get('plan', []):
    st.setdefault('plan', []).append(step)
if step not in st.get('completed', []):
    st.setdefault('completed', []).append(step)
st['next_index'] = max(int(st.get('next_index') or 0), len(st.get('plan', [])))
st['steps_since_summary'] = int(st.get('steps_since_summary') or 0) + 1
st['status'] = 'ready_for_next'
st['updated_utc'] = datetime.now(timezone.utc).isoformat()
st['notes'] = (
    'ds003969 Muse AF7/AF8 (+TP7/TP8) attention blocks ingested. '
    'Next invent: (1) attention_only_head_smoke on ds003969 subject holdout, '
    '(2) more_ds001787_subjects, (3) repack_cbramod_include_pool_head, '
    '(4) personal_muse_cal_blocker (needs_input).'
)
st['step_15_ds003969_attention_ingest'] = {
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'subjects': [s['tag'] for s in summary['subjects']],
    'groups': {s['tag']: s['group'] for s in summary['subjects']},
    'totals': summary['totals'],
    'n_windows_total': summary['n_windows_total'],
    'tasks': summary.get('tasks'),
    'channel_proxy': 'AF7/AF8/TP7/TP8→AF7/AF8/TP9/TP10',
    'label_rule': summary.get('label_rule'),
    'kaggle_private_versioned': [
        'windwerfer/muse-eeg-heads-windows',
        'windwerfer/muse-eeg-heads-cache',
    ],
    'takeaway': (
        'Muse-proximal attention windows from CC0 ds003969; block labels weaker than ds001787 probes '
        'but channels match Muse AF7/AF8. Prefer subject holdout smoke before joint 4-way again.'
    ),
    'artifacts': {
        'script': 'scripts/ds003969_attention_ingest.py',
        'docs': 'docs/ds003969_attention_ingest.md',
        'summary': 'exports/ds003969_attention_ingest_summary.json',
        'provenance': 'kaggle_datasets/muse-eeg-heads-cache/data/ds003969/provenance_attention_ingest.json',
        'exports': 'exports/windows_ds003969/',
    },
}
p.write_text(json.dumps(st, indent=2) + '\n')
print('overnight_state updated', st['status'], 'steps_since_summary', st['steps_since_summary'])
PY

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
