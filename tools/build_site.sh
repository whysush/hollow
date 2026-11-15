#!/bin/bash
# Assemble the static site: site/ = web/* + models/hollow/ (from the given model folder).
# Usage: tools/build_site.sh <model-folder containing config.json + onnx/> [out=site]
set -e
cd /media/sushanth/Elements/hollow_ws
model=$1; out=${2:-site}
rm -rf "$out"; mkdir -p "$out/models"
cp web/*.html web/*.css web/*.js web/*.json "$out/"
[ -d web/assets ] && cp -r web/assets "$out/"   # face source image
cp -r "$model" "$out/models/hollow"
find "$out" -type f -size +95M -print -exec false {} + || { echo "file over 95 MB"; exit 1; }
du -sh "$out"; find "$out" -type f | sort
