#!/usr/bin/env python3
"""Render the bimanual twin at raiden home through its three rig cameras, and optionally run the hold-home
physics smoke test.

    MUJOCO_GL=egl python scripts/raiden_bimanual/render_home.py --out DIR [--hold 5] [--grip]

Renders as scripts/raiden_sim_server.py does: its own mujoco.Renderer, visual geoms (group 1) only, no sites, and
no robosuite offscreen renderer (robosuite's forced camera observations came back as garbage after run_phase
stepping, 09-30; the sim server never uses them). Writes DIR/<camera>.png (top row first), DIR/<camera>_seg.npy
(geom id per pixel, -1 = none), DIR/geoms.json (geom id -> name, body), DIR/poses.json (camera and link_6 world poses) and, with
--hold, DIR/hold.json (joint drift per arm) and DIR/after_hold/ (the same renders after stepping).
Home = all arm joints 0 (encoder zero), fingers open (0.0475 m), as raiden's FOLLOWER_HOME_POS.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import mesa  # noqa: E402
import mesa.sim.controllers  # noqa: E402,F401  (registers YAM_MOTOR_PD)
import mesa.sim.envs.bddl_utils as BDDLUtils  # noqa: E402
from mesa.sim.envs import TASK_MAPPING  # noqa: E402
from mesa.sim.envs.rigs import load_rig  # noqa: E402

TASK = os.path.join(os.path.dirname(mesa.__file__), "task_suites/bddl_files/raiden_bimanual/empty_rig/source/000.json")
CAMS = {"scene_camera": "scene_camera", "left_wrist_camera": "robot0_eye_in_hand", "right_wrist_camera": "robot1_eye_in_hand"}
OPEN = 0.0475  # per-finger stroke, fully open


def make_env(task_json, rig):
    cfg = json.load(open(os.path.join(os.path.dirname(mesa.__file__), "config/controllers/yam_motor_pd.json")))
    problem = BDDLUtils.load_problem(task_json)
    return TASK_MAPPING[problem["problem_name"]](
        parsed_problem=problem, controller_configs=[cfg, cfg], has_renderer=False, has_offscreen_renderer=False,
        render_camera=None, use_camera_obs=False, camera_names=[], control_freq=30, ignore_done=True, hard_reset=True)


class RigRenderer:
    """The sim server's camera path: one mujoco.Renderer per resolution, visual group only, no sites."""

    def __init__(self, env, rig):
        self.model = env.sim.model._model
        (w, h), = {tuple(rig["cameras"][c]["resolution"]) for c in CAMS}
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), w)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), h)
        self.rgb = mujoco.Renderer(self.model, height=h, width=w)
        self.seg = mujoco.Renderer(self.model, height=h, width=w)
        self.seg.enable_segmentation_rendering()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1
        self.opt.sitegroup[:] = 0

    def __call__(self, data, camera):
        self.rgb.update_scene(data, camera=camera, scene_option=self.opt)
        img = self.rgb.render().copy()  # top row first, RGB
        self.seg.update_scene(data, camera=camera, scene_option=self.opt)
        s = self.seg.render()
        return img, np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1).astype(np.int32)


def set_home(env, fingers=OPEN):
    sim = env.sim
    for robot in env.robots:
        sim.data.qpos[robot._ref_joint_pos_indexes] = 0.0
        sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
        for j in robot._ref_gripper_joint_pos_indexes["right"]:
            sim.data.qpos[j] = fingers
    sim.forward()


def cam_pose_cv(model, data, name):
    """world_from_camera with OpenCV axes (x right, y down, z forward)."""
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[i].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = data.cam_xpos[i]
    return T


def body_pose(model, data, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    T = np.eye(4)
    T[:3, :3] = data.xmat[i].reshape(3, 3)
    T[:3, 3] = data.xpos[i]
    return T


def render_all(env, rig, out, renderer):
    os.makedirs(out, exist_ok=True)
    model, data = env.sim.model._model, env.sim.data._data
    for real, sim_name in CAMS.items():
        img, gid = renderer(data, sim_name)
        cv2.imwrite(os.path.join(out, f"{real}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(out, f"{real}_seg.npy"), gid)
    geoms = {int(i): dict(name=model.geom(i).name, body=model.body(model.geom_bodyid[i]).name) for i in range(model.ngeom)}
    json.dump(geoms, open(os.path.join(out, "geoms.json"), "w"))
    poses = dict(cameras={real: cam_pose_cv(model, data, sim_name).tolist() for real, sim_name in CAMS.items()},
                 link_6={r.robot_model.naming_prefix: body_pose(model, data, f"{r.robot_model.naming_prefix}link_6").tolist()
                         for r in env.robots},
                 qpos=data.qpos.tolist())
    json.dump(poses, open(os.path.join(out, "poses.json"), "w"), indent=1)


def run_phase(env, targets, grip, seconds, log):
    """Step at the control rate with absolute joint targets and a gripper command per arm (-1 open, +1 close)."""
    n = int(round(seconds * env.control_freq))
    for _ in range(n):
        action = np.concatenate([np.r_[targets[i], grip[i]] for i in range(len(env.robots))])
        env.step(action)
        log.append(dict(q=[env.sim.data.qpos[r._ref_joint_pos_indexes].tolist() for r in env.robots],
                        fingers=[env.sim.data.qpos[r._ref_gripper_joint_pos_indexes["right"]].tolist() for r in env.robots]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json-path", default=TASK)
    ap.add_argument("--rig", default="raiden_bimanual")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", type=float, default=0.0, help="seconds to hold home under the motor loop")
    ap.add_argument("--grip", action="store_true", help="after the hold, close and open each gripper once")
    args = ap.parse_args()
    rig = load_rig(args.rig)
    env = make_env(args.task_json_path, rig)
    env.reset()
    set_home(env)
    renderer = RigRenderer(env, rig)
    render_all(env, rig, args.out, renderer)
    print(f"rendered {list(CAMS)} to {args.out}", flush=True)
    if args.hold <= 0:
        env.close()
        return
    zeros = [np.zeros(6) for _ in env.robots]
    log = []
    run_phase(env, zeros, [-1.0] * len(env.robots), args.hold, log)
    q = np.array([s["q"] for s in log])  # steps, arms, 6
    res = dict(control_hz=env.control_freq, seconds=args.hold, arms={})
    names = ["left", "right"]
    for a in range(q.shape[1]):
        res["arms"][names[a]] = dict(
            final_q_deg=np.degrees(q[-1, a]).round(3).tolist(),
            max_abs_q_deg=np.degrees(np.abs(q[:, a]).max(0)).round(3).tolist(),
            drift_last_second_deg=np.degrees(np.abs(q[-1, a] - q[-env.control_freq, a])).round(4).tolist(),
            final_fingers_m=log[-1]["fingers"][a])
    if args.grip:
        for a, name in enumerate(names):
            glog = []
            for g, secs in ((1.0, 1.5), (-1.0, 1.5)):
                grip = [-1.0] * len(env.robots)
                grip[a] = g
                run_phase(env, zeros, grip, secs, glog)
            f = np.array([s["fingers"][a] for s in glog])  # steps, 2
            n1 = int(round(1.5 * env.control_freq))
            closed, opened = f[n1 - 1], f[-1]
            t_close = next((i for i in range(n1) if f[i, 0] < 0.001 + f[:n1, 0].min()), None)
            log.extend(glog)
            res["arms"][name]["grip"] = dict(closed_fingers_m=closed.round(5).tolist(), reopened_fingers_m=opened.round(5).tolist(),
                                             close_time_s=None if t_close is None else round((t_close + 1) / env.control_freq, 3),
                                             finger_sync_max_mm=round(float(np.abs(f[:, 0] - f[:, 1]).max() * 1000), 3))
        qg = np.array([s["q"] for s in log])
        for a, name in enumerate(names):
            res["arms"][name]["final_q_after_grip_deg"] = np.degrees(qg[-1, a]).round(3).tolist()
            res["arms"][name]["max_abs_q_during_grip_deg"] = np.degrees(np.abs(qg[q.shape[0]:, a]).max(0)).round(3).tolist()
    json.dump(res, open(os.path.join(args.out, "hold.json"), "w"), indent=1)
    np.save(os.path.join(args.out, "hold_q.npy"), np.array([s["q"] for s in log]))
    print(json.dumps(res, indent=1), flush=True)
    render_all(env, rig, os.path.join(args.out, "after_hold"), renderer)
    print("rendered after the hold", flush=True)
    env.close()


if __name__ == "__main__":
    main()
