"""
Minimal sketch: failure-aware curriculum reset loop in raw MuJoCo.

Demonstrates the three primitives your pipeline needs:
  1. Exact state save/restore (mj_getState / mj_setState equivalents)
  2. IK-based failure detection at goal
  3. Yoshikawa manipulability scan backwards through a trajectory

This is deliberately framework-light so the state plumbing is visible.
Layer robosuite/Gymnasium on top once this pattern is confirmed.
"""

from __future__ import annotations

import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Optional


# ----------------------------------------------------------------------------
# 1. State snapshot: everything you need to exactly restore a sim step.
# ----------------------------------------------------------------------------

@dataclass
class SimState:
    """Full MuJoCo state for exact replay.

    mjData has more fields than this, but qpos/qvel/act/time/ctrl is sufficient
    for deterministic restore when the model is unchanged. If you use
    mocap bodies, plugins, or stochastic actuators, extend this.
    """
    qpos: np.ndarray
    qvel: np.ndarray
    act:  np.ndarray
    time: float
    ctrl: np.ndarray
    # Optional: store warm-start data for contact solver determinism
    qacc_warmstart: np.ndarray

    @classmethod
    def capture(cls, data: mujoco.MjData) -> "SimState":
        return cls(
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            act=data.act.copy() if data.act.size else np.zeros(0),
            time=float(data.time),
            ctrl=data.ctrl.copy(),
            qacc_warmstart=data.qacc_warmstart.copy(),
        )

    def restore(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        if self.act.size:
            data.act[:] = self.act
        data.time = self.time
        data.ctrl[:] = self.ctrl
        data.qacc_warmstart[:] = self.qacc_warmstart
        # forward() propagates kinematics/dynamics without integrating
        mujoco.mj_forward(model, data)


# ----------------------------------------------------------------------------
# 2. Manipulability: Yoshikawa measure at the end-effector for the goal task.
# ----------------------------------------------------------------------------

def yoshikawa_manipulability(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_body_id: int,
    arm_dof_ids: np.ndarray,
) -> float:
    """sqrt(det(J J^T)) for the translational+rotational EEF Jacobian
    restricted to the controllable arm DOFs.

    For a non-redundant 6-DOF arm this reduces to |det(J)|.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
    J = np.vstack([jacp, jacr])[:, arm_dof_ids]  # (6, n_arm)
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return float(np.sqrt(max(det, 0.0)))


# ----------------------------------------------------------------------------
# 3. IK feasibility check at goal, given current object-EEF relative pose.
# ----------------------------------------------------------------------------

def ik_feasible_at_goal(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_body_id: int,
    obj_body_id: int,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    arm_dof_ids: np.ndarray,
    max_iters: int = 50,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-2,
) -> bool:
    """Damped-least-squares IK attempt: can the arm reach a pose that places
    the *object* at the goal, preserving current object-EEF relative pose?

    Returns True if IK converges within tolerance without joint-limit violation.
    This is the 'no IK solution = failure' check from your pipeline.
    """
    # Compute desired EEF pose = goal pose composed with inverse of current
    # object-relative-to-EEF transform. (Omitted for brevity; use mju_mulPose /
    # mju_negPose or your preferred pose math.)
    target_pos, target_quat = compose_target_eef_pose(
        model, data, ee_body_id, obj_body_id, goal_pos, goal_quat
    )

    snapshot = SimState.capture(data)  # don't clobber live sim
    try:
        for _ in range(max_iters):
            mujoco.mj_forward(model, data)
            ee_pos = data.xpos[ee_body_id].copy()
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, data.xmat[ee_body_id])

            pos_err = target_pos - ee_pos
            rot_err = np.zeros(3)
            mujoco.mju_subQuat(rot_err, target_quat, ee_quat)

            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
                # Joint-limit check
                lo = model.jnt_range[arm_dof_ids, 0]
                hi = model.jnt_range[arm_dof_ids, 1]
                q = data.qpos[arm_dof_ids]
                return bool(np.all(q >= lo) and np.all(q <= hi))

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
            J = np.vstack([jacp, jacr])[:, arm_dof_ids]
            err = np.concatenate([pos_err, rot_err])

            # Damped least squares
            damping = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            data.qpos[arm_dof_ids] += 0.5 * dq  # step size
        return False
    finally:
        snapshot.restore(model, data)


def compose_target_eef_pose(model, data, ee_body_id, obj_body_id, goal_pos, goal_quat):
    """Placeholder — compute EEF pose that places object at (goal_pos, goal_quat)
    given current object-EEF relative transform. Use mju_mulPose/mju_negPose."""
    raise NotImplementedError("Fill in with your pose composition.")


# ----------------------------------------------------------------------------
# 4. The curriculum loop: rollout -> detect failure -> scan back -> add rescue.
# ----------------------------------------------------------------------------

class FailureAwareCurriculum:
    def __init__(
        self,
        model: mujoco.MjModel,
        manipulability_threshold: float,
        max_buffer_size: int = 1000,
    ):
        self.model = model
        self.threshold = manipulability_threshold
        self.buffer: list[SimState] = []
        self.max_buffer_size = max_buffer_size

    def sample_initial_state(self, rng: np.random.Generator) -> Optional[SimState]:
        """Return a rescue state, or None to use the default initial distribution.
        Simple uniform mixing; swap for prioritized sampling later."""
        if not self.buffer or rng.random() < 0.5:
            return None
        return rng.choice(self.buffer)

    def process_trajectory(
        self,
        trajectory: list[SimState],
        failed: bool,
        ee_body_id: int,
        arm_dof_ids: np.ndarray,
        data: mujoco.MjData,
    ) -> Optional[SimState]:
        """Scan backwards from failure point; add first state above
        manipulability threshold to the curriculum buffer."""
        if not failed:
            return None

        for state in reversed(trajectory):
            state.restore(self.model, data)
            w = yoshikawa_manipulability(self.model, data, ee_body_id, arm_dof_ids)
            # TODO: also check stability condition (contacted-object velocity
            # in EEF frame, transitively through contact chain)
            if w > self.threshold:
                self._add(state)
                return state
        return None

    def _add(self, state: SimState) -> None:
        self.buffer.append(state)
        if len(self.buffer) > self.max_buffer_size:
            # FIFO; revisit once you evaluate staleness empirically
            self.buffer.pop(0)


# ----------------------------------------------------------------------------
# 5. One training episode, glued together.
# ----------------------------------------------------------------------------

def run_episode(
    model, data, policy, curriculum: FailureAwareCurriculum,
    ee_body_id, obj_body_id, arm_dof_ids, goal_pos, goal_quat,
    max_steps: int, rng: np.random.Generator,
):
    # --- Reset: either default init or a rescue state ---
    rescue = curriculum.sample_initial_state(rng)
    if rescue is not None:
        rescue.restore(model, data)
    else:
        mujoco.mj_resetData(model, data)
        # ... your usual domain init here (object placement, arm home, etc.)
        mujoco.mj_forward(model, data)

    trajectory: list[SimState] = []
    failed = False

    for t in range(max_steps):
        trajectory.append(SimState.capture(data))

        obs = extract_obs(model, data)  # your obs fn
        action = policy(obs)
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        # Failure check: goal unreachable from current (object, EEF) relative pose
        if not ik_feasible_at_goal(
            model, data, ee_body_id, obj_body_id,
            goal_pos, goal_quat, arm_dof_ids,
        ):
            failed = True
            break

        if goal_reached(model, data, obj_body_id, goal_pos, goal_quat):
            break

    # --- Curriculum update ---
    curriculum.process_trajectory(
        trajectory, failed, ee_body_id, arm_dof_ids, data,
    )

    return trajectory, failed


def extract_obs(model, data): ...
def goal_reached(model, data, obj_body_id, goal_pos, goal_quat): ...


if __name__ == "__main__":
    # Load any MuJoCo model — grab one from mujoco_menagerie
    model = mujoco.MjModel.from_xml_path(
        "mujoco_menagerie/franka_emika_panda/panda.xml"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Capture, perturb, restore
    snap = SimState.capture(data)
    print(f"Captured qpos[:3]: {data.qpos[:3]}")

    for _ in range(100):
        data.ctrl[:] = np.random.randn(model.nu) * 0.1
        mujoco.mj_step(model, data)
    print(f"After 100 steps qpos[:3]: {data.qpos[:3]}")

    snap.restore(model, data)
    print(f"After restore qpos[:3]: {data.qpos[:3]}")
    assert np.allclose(data.qpos, snap.qpos), "State restore failed!"
    print("✓ State round-trip works")
