#!/bin/bash
# Resumable raw-data downloads (curl -C - continues partial files after a disconnect).
source /media/sushanth/SSD/hollow_ws/env.sh
cd "$PROJECT_ROOT/data/raw"
HF=https://huggingface.co/datasets
for i in 00000 00001; do
  f=wikipedia/train-$i-of-00041.parquet
  until curl -sSL -C - --retry 5 -o $f.part $HF/wikimedia/wikipedia/resolve/main/20231101.en/train-$i-of-00041.parquet; do sleep 10; done
  mv $f.part $f; echo "done $f"
done
for i in 000 001; do
  f=fineweb/${i}_00000.parquet
  until curl -sSL -C - --retry 5 -o $f.part $HF/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/${i}_00000.parquet; do sleep 10; done
  mv $f.part $f; echo "done $f"
done
echo DL_DONE
