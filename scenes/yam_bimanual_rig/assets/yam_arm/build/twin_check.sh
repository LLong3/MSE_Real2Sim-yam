#!/bin/bash
# Check render of the twin with other wrist cameras / arm asset (09-30 wrist/claw fit). Background Blender, one GPU job.
# usage: build/twin_check.sh DIR ARMS_BLEND [VIEWS] [SAMPLES]
#   DIR/config.json (+ optional DIR/metric_frame.json) from build/make_twin_config.py. Writes DIR/twin.blend,
#   DIR/renders/ (render_views.py), compare_<view>/ (compare.py), claw_check.json (claw_check.py); logs in DIR.
#   The build scripts in scenes/yam_bimanual_rig/build are used read only.
set -e
DIR=$1; ARMS=$2; VIEWS=${3:-scene,left_wrist,right_wrist}; S=${4:-256}
B=/home/frank/robot/aha-3d/scenes/yam_bimanual_rig/build
PY=/home/frank/robot/raiden/.venv/bin/python
source /home/frank/robot/aha-3d/kimodo_blender/env.sh >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1
METRIC_ARG=""; [ -f "$DIR/metric_frame.json" ] && METRIC_ARG="--metric $DIR/metric_frame.json"
"$BLENDER_BIN" -b --factory-startup --python-exit-code 1 --python $B/build_scene.py -- --config $DIR/config.json $METRIC_ARG \
    --out $DIR/twin.blend --arms $ARMS --arms-status final > $DIR/build.log 2>&1
"$BLENDER_BIN" -b $DIR/twin.blend --python-exit-code 1 --python $B/render_views.py -- --out $DIR/renders --views $VIEWS --samples $S > $DIR/render.log 2>&1
cd $B
for v in ${VIEWS//,/ }; do
  $PY compare.py --render $DIR/renders --config $DIR/config.json --view $v --out $DIR/renders/compare_$v > $DIR/compare_$v.log 2>&1
done
if [[ "$VIEWS" == *scene* && "$VIEWS" == *left_wrist* && "$VIEWS" == *right_wrist* ]]; then
  $PY claw_check.py $DIR/renders $DIR/config.json $DIR/renders/claw_check.json > $DIR/claw_check.log 2>&1
fi
grep -h "BUILD_OK\|RENDER_OK" $DIR/build.log $DIR/render.log
echo TWIN_CHECK_OK
