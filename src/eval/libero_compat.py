from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "glx"

import mujoco
from robosuite.utils import binding_utils

assert binding_utils.GLContext.__name__ == "GLFWGLContext", (
    f"expected the GLFW context, got {binding_utils.GLContext.__name__}; "
    "EGL/OSMesa have no working device on WSL2"
)

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)


def _addr(self, name, qpos: bool):
    jid = self.joint_name2id(name)
    jtype = int(self.jnt_type[jid])
    start = int(self.jnt_qposadr[jid] if qpos else self.jnt_dofadr[jid])
    if jtype == _FREE:
        ndim = 7 if qpos else 6
    elif jtype == _BALL:
        ndim = 4 if qpos else 3
    else:
        assert jtype in (_HINGE, _SLIDE), f"unexpected joint type {jtype}"
        ndim = 1
    return start if ndim == 1 else (start, start + ndim)


binding_utils.MjModel.get_joint_qpos_addr = lambda self, name: _addr(self, name, True)
binding_utils.MjModel.get_joint_qvel_addr = lambda self, name: _addr(self, name, False)
