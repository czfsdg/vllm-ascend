#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-click D-Cut claim verifier.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"

MODEL="${MODEL:-/data/wenxuan/Qwen3.5-9B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/wenxuan/Qwen3.5-9B-DFlash}"
DATASET="${DATASET:-/data/wenxuan/speculators/data/allava/allava_10000.jsonl}"
TP="${TP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
SPEC_TOKENS="${SPEC_TOKENS:-15}"
SEQLEN="${SEQLEN:-4096}"
TAIL_FRAC="${TAIL_FRAC:-0.1}"
THROUGHPUT_CONC="${THROUGHPUT_CONC:-16,32,64}"
NUM="${NUM:-256}"
WARMUP="${WARMUP:-16}"
MAXTOK="${MAXTOK:-128}"
REPEATS="${REPEATS:-3}"
LOSSLESS_CONC="${LOSSLESS_CONC:-1}"
LOSSLESS_NUM="${LOSSLESS_NUM:-64}"
LOSSLESS_MAXTOK="${LOSSLESS_MAXTOK:-256}"
OUT_ROOT="${OUT_ROOT:-/data/wenxuan/dcut_verify/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"

DCUT_CONFIG="$OUT_ROOT/dcut_config.json"
cat > "$DCUT_CONFIG" <<JSON
{
  "enabled": true,
  "warmup_batch_sizes": [],
  "min_warmup_batch_size": 2,
  "max_warmup_batch_size": null,
  "min_query_len_per_req": 2,
  "max_query_len_per_req": null,
  "query_len_step_per_req": 2,
  "warmup_seq_lens": $SEQLEN,
  "n_warmup_iters": 3,
  "n_measure_iters": 5
}
JSON

python3 - <<'PY'
from dcut.verify_adaptive_controller import choose_query_lens_discrete
r = choose_query_lens_discrete(
    probs=[[0.9, 0.8], [0.5, 0.5]],
    base_batch_size=2,
    q_levels=[2, 4, 6],
    cost_lookup=lambda q: {2: 1.0, 4: 1.2, 6: 2.0}[q],
    max_draft_len=2,
)
assert r["draft_lens"] == [2, 0] and r["best_Q"] == 4, f"FAIL controller: {r}"
print(" OK: controller algorithm sane (draft_lens=[2,0], best_Q=4)")
PY

run_ab() {
  local out_dir="$1" concs="$2" num="$3" maxtok="$4" warmup="$5" variants="${6:-vanilla,dcut}"
  MODEL="$MODEL" DRAFT_MODEL="$DRAFT_MODEL" DATASET="$DATASET" \
  TP="$TP" MAX_NUM_SEQS="$MAX_NUM_SEQS" SPEC_TOKENS="$SPEC_TOKENS" \
  DCUT_CONFIG="$DCUT_CONFIG" \
  VLLM_LOGGING_LEVEL=INFO \
  EXTRA_SERVE_ARGS="--max-model-len $SEQLEN --no-async-scheduling" \
  BENCH_EXTRA_ARGS="--temperature 0.0" \
  VARIANTS="$variants" \
  CONCURRENCY_LIST="$concs" \
  NUM="$num" WARMUP="$warmup" TAIL_FRAC="$TAIL_FRAC" MAXTOK="$maxtok" \
  OUT_DIR="$out_dir" \
  bash "$SCRIPT_DIR/run_qwen35_dcut_ab.sh"
}

echo "### Step 1 lossless phase (c$LOSSLESS_CONC, temp 0, deterministic)"
run_ab "$OUT_ROOT/lossless" "$LOSSLESS_CONC" "$LOSSLESS_NUM" "$LOSSLESS_MAXTOK" 4

echo "### Step 1b determinism control (vanilla vs vanilla, c$LOSSLESS_CONC/temp0)"
run_ab "$OUT_ROOT/controlA" "$LOSSLESS_CONC" "$LOSSLESS_NUM" "$LOSSLESS_MAXTOK" 4 "vanilla"
run_ab "$OUT_ROOT/controlB" "$LOSSLESS_CONC" "$LOSSLESS_NUM" "$LOSSLESS_MAXTOK" 4 "vanilla"

echo "### Step 2 throughput phase (c=$THROUGHPUT_CONC x$REPEATS)"
for i in $(seq 1 "$REPEATS"); do
  echo "--- repeat $i/$REPEATS"
  run_ab "$OUT_ROOT/rep$i" "$THROUGHPUT_CONC" "$NUM" "$MAXTOK" "$WARMUP"
done

echo "### Step 3 claim=ACTIVE"
active_ok=1
cost_table="$OUT_ROOT/lossless/dcut_cost_table.json"
if [[ -s "$cost_table" ]]; then
  entries="$(python3 -c "import json;print(len(json.load(open('$cost_table'))['cost_table']))" 2>/dev/null || echo 0)"
  if [[ "${entries:-0}" -ge 1 ]]; then
    echo " OK cost table exported: $entries entries"
  else
    echo " FAIL cost table is empty"; active_ok=0
  fi
else
  echo " FAIL no cost table at $cost_table"; active_ok=0
fi

echo "### Step 4 claim=LOSSLESS"
lossless_ok=1
python3 - "$OUT_ROOT" "$LOSSLESS_CONC" <<'PY' || lossless_ok=0
import json
import sys
root, conc = sys.argv[1], sys.argv[2]

def load(p):
    try:
        f = open(p, encoding="utf-8")
    except FileNotFoundError:
        return None
    out = {}
    for line in f:
        r = json.loads(line)
        if r.get("ok"):
            out[r["sample_id"]] = (
                r.get("completion_tokens"), r.get("output_chars"), r.get("output_preview"))
    return out

def cmp(a, b):
    if a is None or b is None:
        return None
    keys = set(a) & set(b)
    return (sum(a[k] == b[k] for k in keys), len(keys)) if keys else (0, 0)

ctrl = cmp(load(f"{root}/controlA/vanilla_c{conc}.jsonl"),
           load(f"{root}/controlB/vanilla_c{conc}.jsonl"))
treat = cmp(load(f"{root}/lossless/vanilla_c{conc}.jsonl"),
            load(f"{root}/lossless/dcut_c{conc}.jsonl"))
if treat is None:
    print(" FAIL: missing lossless arm files")
    raise SystemExit(1)
ts, tn = treat
print(f" treatment (vanilla vs dcut) : identical {ts}/{tn}")
if ctrl is None:
    raise SystemExit(0 if ts == tn else 1)
cs, cn = ctrl
print(f" control (vanilla vs vanilla) : identical {cs}/{cn}")
raise SystemExit(0 if (ts / tn if tn else 1.0) >= (cs / cn if cn else 1.0) else 1)
PY

echo "### Step 5 claim=REAL"
python3 - "$OUT_ROOT" "$REPEATS" <<'PY'
import glob
import json
import statistics as st
import sys
root, reps = sys.argv[1], int(sys.argv[2])
agg = {}
for i in range(1, reps + 1):
    for p in glob.glob(f"{root}/rep{i}/*.summary.json"):
        s = json.load(open(p, encoding="utf-8"))
        tag = s.get("tag", "")
        if "_c" not in tag:
            continue
        variant, conc = tag.rsplit("_c", 1)
        agg.setdefault((variant, int(conc)), []).append(s)
print(f" {'conc':>5} {'vanilla out tok/s':>18} {'dcut out tok/s':>15} {'delta':>8}")
for c in sorted({c for _, c in agg}):
    va = agg.get(("vanilla", c), [])
    da = agg.get(("dcut", c), [])
    if not va or not da:
        continue
    def mean(rows, key):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return st.mean(xs) if xs else None
    v_ot, d_ot = mean(va, "output_tok_per_s"), mean(da, "output_tok_per_s")
    delta = f"{(d_ot / v_ot - 1) * 100:+.1f}%" if v_ot and d_ot else "NA"
    print(f" {c:>5} {v_ot or float('nan'):>18.1f} {d_ot or float('nan'):>15.1f} {delta:>8}")
PY

echo "==================================================================="
echo " VERDICT"
[[ "$active_ok" == 1 ]] && echo " ACTIVE : PASS" || echo " ACTIVE : FAIL"
[[ "$lossless_ok" == 1 ]] && echo " LOSSLESS : PASS" || echo " LOSSLESS : FAIL"
echo " REAL : see Step 5 mean delta above"
echo " artifacts: $OUT_ROOT"
echo "==================================================================="
