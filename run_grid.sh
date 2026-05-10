#!/bin/bash
# Grid test

MODEL="gpt-5.4"
REPEAT=5
APPS=(1 2 3 4 5 6 7 8)

echo "Grid Test: model=$MODEL repeat=$REPEAT apps=${APPS[*]}"
date
echo ""

for app in "${APPS[@]}"; do
  echo "=== App $app ==="

  # echo "  [LLM] starting..."
  # python main.py --app "$app" --repeat "$REPEAT" --mode default --model "$MODEL" 2>&1 | tail -8

  echo "  [KG t=1.0] starting..."
  python main_kg.py --app "$app" --repeat "$REPEAT" --dedup-threshold 1.0 --mode kg --model "$MODEL" 2>&1 | tail -8

  echo "  [KG t=0.85] starting..."
  python main_kg.py --app "$app" --repeat "$REPEAT" --dedup-threshold 0.85 --mode kg --model "$MODEL" 2>&1 | tail -8

  echo "  App $app done"
  echo ""
done
date
