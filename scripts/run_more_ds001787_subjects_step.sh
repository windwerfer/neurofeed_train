#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/box/.local/bin:/usr/bin:$PATH"
ROOT=/workspace/muse-eeg-heads
cd "$ROOT"
source .venv/bin/activate

LOG="$ROOT/exports/more_ds001787_subjects_run.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

python scripts/more_ds001787_subjects.py

python - <<'PY'
from pathlib import Path
import json
from datetime import datetime, timezone
root = Path('/workspace/muse-eeg-heads')
summary = json.loads((root/'exports/more_ds001787_subjects_summary.json').read_text())
rows = []
for s in summary['subjects']:
    c = s['counts']
    rows.append(f"| {s['tag']} | {s['group']} | {c.get('concentration',0)} | {c.get('mind_wandering',0)} | {s['n_windows_total']} | {s.get('merge_mode','')} |")
doc = f"""# more_ds001787_subjects (attention pool expand)

**Step:** `more_ds001787_subjects`  
**Completed (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}  
**Dataset:** OpenNeuro `ds001787` — SPDX **CC0-1.0** — DOI `doi:10.18112/openneuro.ds001787.v1.1.1`

## Why

Prior attention-only / joint holdouts collapsed (~chance). Add **2 expert + 2 novice** ses-01 recordings to the Muse-proxy attention pool before any retrain.

## New subjects

| Tag | Group | concentration | mind_wandering | total | merge |
|-----|-------|--------------:|---------------:|------:|-------|
{chr(10).join(rows)}
| **new total** | | **{summary['totals_new'].get('concentration',0)}** | **{summary['totals_new'].get('mind_wandering',0)}** | **{summary['n_windows_new']}** | |

**Pool now:** {', '.join(summary['pool_subjects'])}  
**Pool windows:** concentration={summary['pool_totals'].get('concentration',0)}, mind_wandering={summary['pool_totals'].get('mind_wandering',0)}, total={summary['n_windows_pool']}

## Label / channel rules (unchanged)

- Q1>Q2 → concentration; Q1<Q2 → mind_wandering; ties dropped
- **Hard reject:** do not map events value 2/4 to classes
- Channels: AF7/AF8 + P9/P10→TP9/TP10 (BioSemi64)

## Artifacts

- Script: `scripts/more_ds001787_subjects.py`
- Exports: `exports/windows_ds001787/` (new tags)
- Summary: `exports/more_ds001787_subjects_summary.json`
- Provenance: `kaggle_datasets/muse-eeg-heads-cache/data/ds001787/provenance_more_ds001787_subjects.json`

## Out of scope

- No Head A retrain this step (next invent candidate: attention retrain with subject holdout on expanded pool)
- Head B / Head C training untouched
"""
(root/'docs/more_ds001787_subjects.md').write_text(doc)
print('wrote docs')
PY

sleep 5
kaggle datasets version -p "$ROOT/kaggle_datasets/muse-eeg-heads-windows" -m "ds001787 +sub003/004/014/019 attention windows" -r zip 2>&1 || echo "WINDOWS_VERSION_FAIL=$?"
sleep 8
kaggle datasets version -p "$ROOT/kaggle_datasets/muse-eeg-heads-cache" -m "cache ds001787 raw sub003/004/014/019 + provenance" -r zip 2>&1 || echo "CACHE_VERSION_FAIL=$?"

python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path('/workspace/muse-eeg-heads/overnight_state.json')
st = json.loads(p.read_text())
summary = json.loads(Path('/workspace/muse-eeg-heads/exports/more_ds001787_subjects_summary.json').read_text())
step = 'more_ds001787_subjects'
if step not in st.get('plan', []):
    st.setdefault('plan', []).append(step)
if step not in st.get('completed', []):
    st.setdefault('completed', []).append(step)
st['next_index'] = max(int(st.get('next_index') or 0), len(st.get('plan', [])))
st['steps_since_summary'] = int(st.get('steps_since_summary') or 0) + 1
st['status'] = 'ready_for_next'
st['updated_utc'] = datetime.now(timezone.utc).isoformat()
st['notes'] = (
    'more_ds001787_subjects done: +sub003/004/014/019; pool now 8 subjects. '
    'Next invent: (1) attention_retrain_expanded_holdout, '
    '(2) repack_cbramod_include_pool_head, '
    '(3) more_ds003969_subjects, '
    '(4) personal_muse_cal_blocker (needs_input).'
)
st['step_17_more_ds001787_subjects'] = {
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'subjects_new': [s['tag'] for s in summary['subjects']],
    'groups': {s['tag']: s['group'] for s in summary['subjects']},
    'totals_new': summary['totals_new'],
    'n_windows_new': summary['n_windows_new'],
    'pool_subjects': summary['pool_subjects'],
    'pool_totals': summary['pool_totals'],
    'n_windows_pool': summary['n_windows_pool'],
    'channel_proxy': 'AF7/AF8/TP9/TP10 with TP9/TP10←P9/P10 from BioSemi64',
    'label_rule': summary.get('label_rule'),
    'hard_reject': summary.get('hard_reject'),
    'kaggle_private_versioned': [
        'windwerfer/muse-eeg-heads-windows',
        'windwerfer/muse-eeg-heads-cache',
    ],
    'takeaway': (
        'Expanded ds001787 attention pool to 8 subjects (4 expert / 4 novice). '
        'Next: subject-holdout attention retrain before shipping any attention head.'
    ),
    'artifacts': {
        'script': 'scripts/more_ds001787_subjects.py',
        'docs': 'docs/more_ds001787_subjects.md',
        'summary': 'exports/more_ds001787_subjects_summary.json',
        'provenance': 'kaggle_datasets/muse-eeg-heads-cache/data/ds001787/provenance_more_ds001787_subjects.json',
        'exports': 'exports/windows_ds001787/',
    },
}
p.write_text(json.dumps(st, indent=2) + '\n')
print('overnight_state updated', st['status'], 'steps_since_summary', st['steps_since_summary'])
PY

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
