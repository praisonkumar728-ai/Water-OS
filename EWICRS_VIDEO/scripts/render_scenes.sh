#!/usr/bin/env bash
# Render every scene to assets/generated_scenes/*.mp4 (skips scenes already rendered). Usage: scripts/render_scenes.sh [jobs]
cd "$(dirname "$0")/.."
J=${1:-2}
ls scenes/scene_0*.py | xargs -P "$J" -I{} bash -c 'n=$(basename {} .py); [ -f assets/generated_scenes/$n.mp4 ] && { echo "skip $n"; exit 0; }; python3 {} > work/render_$n.log 2>&1 && echo "done $n" || echo "FAILED $n"'
