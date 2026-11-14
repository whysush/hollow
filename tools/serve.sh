#!/bin/bash
# Usage: tools/serve.sh <dir> [port=8765]  -- (re)start a static server; PID kept in .tmp/serve.pid
cd /media/sushanth/Elements/hollow_ws
[ -f .tmp/serve.pid ] && kill $(cat .tmp/serve.pid) 2>/dev/null
for p in $(lsof -t -i :${2:-8765} 2>/dev/null); do kill $p; done
sleep 0.5
nohup python3 -m http.server ${2:-8765} --bind 127.0.0.1 --directory "$1" > .tmp/http.log 2>&1 &
echo $! > .tmp/serve.pid
