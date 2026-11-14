#!/bin/bash
# Sequential, resumable download of the first FineWeb-Edu sample-10BT parquet shards (stream order = file order).
source /media/sushanth/Elements/hollow_ws/env.sh
cd "$PROJECT_ROOT/data/raw/fineweb"
for i in 000 001 002; do
  f=${i}_00000.parquet
  [ -f $f ] && continue
  until curl -sSL -C - --retry 5 -o $f.part https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/$f; do sleep 10; done
  mv $f.part $f; echo "done $f"
done
echo FW_DL_DONE
