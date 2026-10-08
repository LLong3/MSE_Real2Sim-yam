"""The vial-capping success check (``screwed_on cap_0 vial_0``) on states set by hand, no stepping.

    PYTHONPATH=. python -m pytest scripts/raiden_bimanual/test_vial_capping.py
"""

import json
import math
import os

import mujoco
import numpy as np
import pytest

import mesa
import mesa.sim.controllers  # noqa: F401  (registers YAM_MOTOR_PD)
import mesa.sim.envs.bddl_utils as BDDLUtils
from mesa.sim.envs import TASK_MAPPING
from mesa.sim.envs.objects.raiden_bimanual_objects import VIAL_CAPPING

TASK = os.path.join(os.path.dirname(mesa.__file__), "task_suites/bddl_files/raiden_bimanual/vial_capping/source/000.json")
TIGHT = 2 * math.pi * VIAL_CAPPING["cap"]["turns_to_tight"]
DESCENT = VIAL_CAPPING["cap"]["descent"]


@pytest.fixture(scope="module")
def env():
    cfg = json.load(open(os.path.join(os.path.dirname(mesa.__file__), "config/controllers/yam_motor_pd.json")))
    problem = BDDLUtils.load_problem(TASK)
    env = TASK_MAPPING[problem["problem_name"]](
        parsed_problem=problem, controller_configs=[cfg, cfg], has_renderer=False, has_offscreen_renderer=False,
        render_camera=None, use_camera_obs=False, camera_names=[], control_freq=30, ignore_done=True)
    env.reset()
    return env


def _quat(axis, deg):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), math.radians(deg))
    return q


def _mul(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, a, b)
    return q


def place(env, thread=TIGHT, tilt=0.0, off_axis=0.0, lift=0.0, yaw=30.0, vial_tilt=0.0):
    """Stand the vial at its start, tilted ``vial_tilt`` deg about x; set the cap on its neck, in the vial's frame."""
    m, d = env.sim.model._model, env.sim.data._data
    vial, cap = env.objects_dict["vial_0"], env.objects_dict["cap_0"]
    va = m.jnt_qposadr[m.joint(vial.joints[-1]).id]
    ca = m.jnt_qposadr[m.joint(cap.joints[-1]).id]
    qv = _quat([1, 0, 0], vial_tilt)
    d.qpos[va + 3:va + 7] = qv
    mujoco.mj_forward(m, d)
    R = d.xmat[m.body(vial.root_body).id].reshape(3, 3)
    d.qpos[ca:ca + 3] = d.xpos[m.body(vial.root_body).id] + R @ [off_axis, 0, vial.rim_height + cap.seated_height + lift]
    d.qpos[ca + 3:ca + 7] = _mul(qv, _mul(_quat([0, 1, 0], tilt), _quat([0, 0, 1], yaw)))
    d.qpos[m.jnt_qposadr[m.joint(cap.thread_joint).id]] = thread
    mujoco.mj_forward(m, d)
    return bool(env._check_success())


@pytest.mark.parametrize("kw", [{}, {"yaw": 200.0}, {"thread": TIGHT - math.radians(5)}, {"vial_tilt": 20.0},
                                {"tilt": 4.0}, {"off_axis": 0.0015}, {"lift": 0.002}, {"lift": 0.0025}])
def test_screwed_on(env, kw):
    assert place(env, **kw)


# the first two as the thread leaves them: set on and not turned, and turned half-way
@pytest.mark.parametrize("kw", [{"thread": 0.0, "lift": DESCENT}, {"thread": TIGHT / 2, "lift": DESCENT / 2},
                                {"thread": 0.0}, {"thread": math.pi}, {"thread": TIGHT - math.radians(15)},
                                {"tilt": 10.0}, {"off_axis": 0.003}, {"lift": DESCENT}, {"lift": 0.03},
                                {"vial_tilt": 20.0, "tilt": -10.0}])
def test_not_screwed_on(env, kw):
    assert not place(env, **kw)
