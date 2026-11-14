#!/bin/bash
# Usage: tools/run_bg.sh <logname> <memmax e.g. 3G> <command...>
# Runs a command in the background with env.sh sourced, logging to logs/<logname>.log, inside its own systemd user
# scope with a hard memory cap. Reason: on 2026-11-14 16:18 a global OOM (15 GB RAM shared with the desktop) killed
# dbus and with it the whole desktop session and every job. With MemoryMax, only the offending job is killed.
source /media/sushanth/Elements/hollow_ws/env.sh
cd "$PROJECT_ROOT"
name=$1; mem=$2; shift 2
nohup systemd-run --user --scope -q -p MemoryMax=$mem -p MemorySwapMax=0 --unit="hollow-$name-$(date +%s)" "$@" >> "logs/$name.log" 2>&1 &
echo "started $name (MemoryMax=$mem) pid $!"
