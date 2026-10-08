#!/bin/bash
# usage: iterate.sh VER [views] [samples]  -> build configs/VER.json into the run's iter/VER.blend and render it
set -e
cd "$(dirname "$0")"
source ../../../kimodo_blender/env.sh >/dev/null 2>&1
RUN=/home/frank/robot/aha-3d/runs/yam_bimanual_rig/20260929t225835z-build-1de8ab34
V=$1; VIEWS=${2:-scene,left_wrist,right_wrist}; S=${3:-64}
mkdir -p $RUN/iter $RUN/renders logs
"$BLENDER_BIN" -b --factory-startup --python-exit-code 1 --python build_scene.py -- --config configs/$V.json --out $RUN/iter/$V.blend --arms-status ${ARMS_STATUS:-placeholder} > logs/build_$V.log 2>&1
"$BLENDER_BIN" -b $RUN/iter/$V.blend --python-exit-code 1 --python render_views.py -- --out $RUN/renders/$V --views $VIEWS --samples $S > logs/render_$V.log 2>&1
grep -h "BUILD_OK\|VIEW_OK\|RENDER_OK" logs/build_$V.log logs/render_$V.log
