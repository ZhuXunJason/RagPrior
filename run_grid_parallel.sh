#!/bin/bash
# Parallel grid test with automatic fallback to sequential on failure
# Apps 1-8, model deepseek-v4-pro, repeat=5

MODEL="deepseek-v4-pro"
REPEAT=5
APPS=(1 2 3 4 5 6 7 8)

echo "Grid Test: model=$MODEL repeat=$REPEAT (parallel→sequential fallback)"
date

for app in "${APPS[@]}"; do
  echo ""
  echo "=== App $app ==="

  # --- Phase 1: Parallel ---
  echo "  [parallel] launching 3 threads..."
  (python main.py --app "$app" --repeat "$REPEAT" --mode default --model "$MODEL" > /tmp/grid_${app}_llm.log 2>&1) &
  (python main_kg.py --app "$app" --repeat "$REPEAT" --dedup-threshold 1.0 --mode kg --model "$MODEL" > /tmp/grid_${app}_kg10.log 2>&1) &
  (python main_kg.py --app "$app" --repeat "$REPEAT" --dedup-threshold 0.85 --mode kg --model "$MODEL" > /tmp/grid_${app}_kg85.log 2>&1) &
  wait

  # --- Check results ---
  missing=()
  for mode in "llm-deepseek-v4-pro" "kg-deepseek-v4-pro-t1.0" "kg-deepseek-v4-pro-t0.85"; do
    if [ ! -f "result/${app}-${mode}-1.txt" ]; then
      missing+=("$mode")
    fi
  done

  if [ ${#missing[@]} -eq 0 ]; then
    echo "  [parallel] all OK"
  else
    echo "  [parallel] ${#missing[@]} failed, re-running sequentially..."
    for mode in "${missing[@]}"; do
      case "$mode" in
        llm-deepseek-v4-pro)
          echo "    re-running LLM..."
          python main.py --app "$app" --repeat "$REPEAT" --mode default --model "$MODEL" > /tmp/grid_${app}_llm.log 2>&1
          ;;
        kg-deepseek-v4-pro-t1.0)
          echo "    re-running KG t=1.0..."
          python main_kg.py --app "$app" --repeat "$REPEAT" --dedup-threshold 1.0 --mode kg --model "$MODEL" > /tmp/grid_${app}_kg10.log 2>&1
          ;;
        kg-deepseek-v4-pro-t0.85)
          echo "    re-running KG t=0.85..."
          python main_kg.py --app "$app" --repeat "$REPEAT" --dedup-threshold 0.85 --mode kg --model "$MODEL" > /tmp/grid_${app}_kg85.log 2>&1
          ;;
      esac
    done
    echo "  [sequential] retry done"
  fi
  echo "  App $app complete"
done

# Build JSON cache
echo ""
echo "=== Building JSON ==="
python -c "
import sys, json, re, numpy as np, os
sys.path.insert(0,'.')
from main import run_evaluate as llm_eval
from main_kg import run_evaluate as kg_eval
import io
from contextlib import redirect_stdout

MODEL = '$MODEL'
REPEAT = $REPEAT
model_tag = MODEL.replace('-', '')

for app in range(1, 9):
    for mode_name, label, fn in [
        (f'llm-{MODEL}', 'llm', llm_eval),
        (f'kg-{MODEL}-t1.0', 'kg_t1.0', kg_eval),
        (f'kg-{MODEL}-t0.85', 'kg_t0.85', kg_eval),
    ]:
        fname = f'cache/{label.split(\"_\")[0]}_app{app}_{model_tag}'
        if label != 'llm':
            fname += f'_{label.split(\"_\")[1]}'
        fname += '.json'

        # Check result files exist
        result_ok = all(os.path.exists(f'result/{app}-{mode_name}-{i}.txt') for i in range(1, REPEAT+1))
        if not result_ok:
            print(f'  {label} app{app}: result files missing, skip')
            continue

        f = io.StringIO()
        with redirect_stdout(f):
            fn(app=app, repeat=REPEAT, mode=mode_name, print_each=True)
        out = f.getvalue()
        if label == 'llm':
            runs = re.findall(r'^\d+\s+([0-9]+\.[0-9]+)', out, re.MULTILINE)
        else:
            runs = re.findall(r'\[(\d+)\]\s+APFD = ([0-9]+\.[0-9]+)', out)
            runs = [r[1] for r in runs]
        apfds = [round(float(r), 4) for r in runs]
        if apfds:
            data = {'runs': apfds, 'mean': round(float(np.mean(apfds)), 4), 'min': round(min(apfds), 4), 'max': round(max(apfds), 4)}
            with open(fname, 'w') as f2:
                json.dump(data, f2, indent=2)
            print(f'  {label} app{app}: mean={data[\"mean\"]:.4f}')
        else:
            print(f'  {label} app{app}: no APFD parsed')
" 2>&1

# Print table
echo ''
echo '========================================================'
echo "  APFD Summary (model=$MODEL, repeat=$REPEAT)"
echo '========================================================'
printf '%-4s  %-12s  %-14s  %-14s\n' 'App' 'LLM' 'KG t=1.0' 'KG t=0.85'
printf '%-4s  %-12s  %-14s  %-14s\n' '---' '------------' '--------------' '--------------'

model_tag=$(echo "$MODEL" | tr -d '-')
sum_llm=0; sum_t10=0; sum_t85=0; c_llm=0; c_t10=0; c_t85=0
for app in "${APPS[@]}"; do
  llm=$(python -c "import json; d=json.load(open('cache/llm_app%d_%s.json' % ($app, '$model_tag'))); print('%.4f' % d['mean'])" 2>/dev/null || echo "N/A")
  t10=$(python -c "import json; d=json.load(open('cache/kg_app%d_%s_t1.0.json' % ($app, '$model_tag'))); print('%.4f' % d['mean'])" 2>/dev/null || echo "N/A")
  t85=$(python -c "import json; d=json.load(open('cache/kg_app%d_%s_t0.85.json' % ($app, '$model_tag'))); print('%.4f' % d['mean'])" 2>/dev/null || echo "N/A")
  printf '%-4s  %-12s  %-14s  %-14s\n' "$app" "$llm" "$t10" "$t85"
  [ "$llm" != "N/A" ] && sum_llm=$(python -c "print($sum_llm + $llm)") && c_llm=$((c_llm+1))
  [ "$t10" != "N/A" ] && sum_t10=$(python -c "print($sum_t10 + $t10)") && c_t10=$((c_t10+1))
  [ "$t85" != "N/A" ] && sum_t85=$(python -c "print($sum_t85 + $t85)") && c_t85=$((c_t85+1))
done
echo '--------------------------------------------------------'
[ $c_llm -gt 0 ] && avg_llm=$(python -c "print(round($sum_llm / $c_llm, 4))") || avg_llm="N/A"
[ $c_t10 -gt 0 ] && avg_t10=$(python -c "print(round($sum_t10 / $c_t10, 4))") || avg_t10="N/A"
[ $c_t85 -gt 0 ] && avg_t85=$(python -c "print(round($sum_t85 / $c_t85, 4))") || avg_t85="N/A"
printf '%-4s  %-12s  %-14s  %-14s\n' 'Avg' "$avg_llm" "$avg_t10" "$avg_t85"
echo '========================================================'
date
