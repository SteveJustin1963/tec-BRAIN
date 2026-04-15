"""
pc_visual_robot.py — Visual Active Inference for a 2-DOF robot arm.

Extends the 1-D active inference in pc_robot.py to 2-D camera space.
A colored marker on the gripper is tracked by a (simulated) camera;
the PC brain drives the arm by minimising the visual prediction error —
no inverse kinematics, no knowledge of arm geometry.

  1. PERCEPTION  — PCStateEstimator2D filters noisy camera readings
                   into a smooth 2-D belief μ = (x̂, ŷ).

  2. ACTION      — ActiveInferenceController2D outputs a Cartesian
                   velocity command that minimises the goal error:
                       u = κ_goal · (goal − μ)

  3. MAPPING     — The arm's Jacobian (known only to the hardware /
                   simulation) maps the velocity command to joint
                   velocities.  The PC brain never touches this.

BASELINE: a plain proportional controller using the raw (noisy) camera
reading is run in parallel so you can see the benefit of PC filtering.

Run: python3 pc_visual_robot.py
"""

import numpy as np
import math

np.random.seed(7)

PIXEL_NOISE = 0.08   # σ of simulated camera noise (meters)
DT          = 0.1    # simulation timestep (seconds)


# ── 1.  Simulated 2-DOF planar arm ────────────────────────────────────────────

class Arm2DOF:
    """
    Two-link planar arm.
      theta1 : shoulder angle (radians, measured from x-axis)
      theta2 : elbow angle relative to upper arm (radians)
      l1, l2 : link lengths (meters)

    Forward kinematics → end-effector (x, y) in the plane.
    The PC brain never uses this geometry; it only sees camera coords.
    """

    def __init__(self, l1=1.0, l2=0.8, theta1=0.3, theta2=0.5):
        self.l1     = l1
        self.l2     = l2
        self.theta1 = theta1
        self.theta2 = theta2

    def fk(self):
        """Return (x, y) Cartesian position of the end-effector."""
        t12 = self.theta1 + self.theta2
        x   = self.l1 * math.cos(self.theta1) + self.l2 * math.cos(t12)
        y   = self.l1 * math.sin(self.theta1) + self.l2 * math.sin(t12)
        return np.array([x, y])

    def _jacobian(self):
        """2×2 Jacobian J such that ẋ = J · θ̇."""
        t12 = self.theta1 + self.theta2
        return np.array([
            [-self.l1 * math.sin(self.theta1) - self.l2 * math.sin(t12),
             -self.l2 * math.sin(t12)],
            [ self.l1 * math.cos(self.theta1) + self.l2 * math.cos(t12),
              self.l2 * math.cos(t12)]
        ])

    def apply_cartesian_velocity(self, v_xy):
        """
        Move the arm by a Cartesian velocity command v_xy.

        Uses damped-least-squares pseudo-inverse of the Jacobian.
        In real hardware this runs on the motor controller firmware;
        the PC brain only produces v_xy — it never calls this directly.
        """
        J     = self._jacobian()
        lam   = 0.01                                        # damping
        Jpinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(2))
        dq    = Jpinv @ v_xy
        self.theta1 += dq[0] * DT
        self.theta2  = float(np.clip(self.theta2 + dq[1] * DT, -2.8, 2.8))

    def camera_reading(self, noise=PIXEL_NOISE):
        """Simulate a noisy camera: true end-effector pos + Gaussian noise."""
        return self.fk() + np.random.randn(2) * noise


# ── 2.  PC state estimator (2-D belief over marker position) ──────────────────

class PCStateEstimator2D:
    """
    2-D Predictive Coding state estimator for a visual marker.

    Generative model:
        belief  μ      : believed (x, y) of marker in camera coords
        prediction     : ŷ = μ              (identity observation model)
        sensory error  : ε_s = y_cam − μ
        prior error    : ε_p = μ − μ₀       (drift from previous belief)

    Free-energy gradient update (iterated, α small step size):
        Δμ = α · (κ_s · ε_s − κ_p · ε_p)

    κ_s and κ_p are precision weights:
        κ_s high → trust camera more (good SNR)
        κ_p high → penalise sudden jumps in belief (smooth motion prior)
    """

    def __init__(self, init_pos,
                 kappa_sensory=3.0,
                 kappa_prior=0.8,
                 steps=20,
                 alpha=0.05):
        self.mu    = np.array(init_pos, dtype=float)
        self.mu0   = self.mu.copy()
        self.k_s   = kappa_sensory
        self.k_p   = kappa_prior
        self.steps = steps
        self.alpha = alpha

    def update(self, y_camera, predicted_pos=None):
        """
        Refine belief given a new camera reading y_camera.
        predicted_pos: dynamics-based prior for this step (optional).
        """
        if predicted_pos is not None:
            self.mu0 = np.array(predicted_pos, dtype=float)
        mu = self.mu.copy()
        for _ in range(self.steps):
            eps_s = y_camera - mu
            eps_p = mu - self.mu0
            mu    = mu + self.alpha * (self.k_s * eps_s - self.k_p * eps_p)
        self.mu0 = self.mu.copy()   # prior for next step = current belief (stationary motion model)
        self.mu  = mu
        return self.mu.copy()


# ── 3.  Active Inference visual controller ────────────────────────────────────

class ActiveInferenceController2D:
    """
    Visual Active Inference motor controller.

    The agent holds a preferred state (goal_xy) in camera coordinates.
    The motor command minimises the visual prediction error:

        u = κ_goal · (goal − μ)

    This controller NEVER uses joint angles or arm geometry.
    It reasons entirely in camera / Cartesian space and outputs
    a velocity vector that the hardware maps to joint commands.

    Biological analogy: efference copy encodes the desired sensory
    state; motor neurons fire to minimise proprioceptive error until
    the predicted sensation matches the desired one.
    """

    def __init__(self, goal_xy, kappa_goal=1.5, max_speed=0.4):
        self.goal  = np.array(goal_xy, dtype=float)
        self.k     = kappa_goal
        self.vmax  = max_speed

    def command(self, believed_pos):
        """Return Cartesian velocity command from visual prediction error."""
        v     = self.k * (self.goal - believed_pos)
        speed = np.linalg.norm(v)
        if speed > self.vmax:
            v = v * (self.vmax / speed)
        return v

    def set_goal(self, new_goal):
        self.goal = np.array(new_goal, dtype=float)


# ── 4.  Classical P-controller baseline (raw noisy camera) ────────────────────

class ProportionalController2D:
    """
    Classical proportional controller using the raw camera reading.
    No noise filtering — used as a baseline to show the benefit of
    the PC state estimator.
    """

    def __init__(self, goal_xy, kp=1.5, max_speed=0.4):
        self.goal = np.array(goal_xy, dtype=float)
        self.kp   = kp
        self.vmax = max_speed

    def command(self, raw_camera_pos):
        v     = self.kp * (self.goal - raw_camera_pos)
        speed = np.linalg.norm(v)
        if speed > self.vmax:
            v = v * (self.vmax / speed)
        return v


# ── 5.  Simulation ────────────────────────────────────────────────────────────

def run_simulation(goal_xy=(0.8, 1.2), steps=120, noise=PIXEL_NOISE):
    goal = np.array(goal_xy, dtype=float)

    # Two arms with identical starting configuration
    arm_pc = Arm2DOF(theta1=0.3, theta2=0.5)
    arm_p  = Arm2DOF(theta1=0.3, theta2=0.5)

    estimator = PCStateEstimator2D(
        init_pos=arm_pc.fk(), kappa_sensory=3.0, kappa_prior=0.8
    )
    pc_ctrl = ActiveInferenceController2D(goal_xy=goal, kappa_goal=1.5)
    p_ctrl  = ProportionalController2D(goal_xy=goal, kp=1.5)

    hist = {
        "pc_true":  [],
        "pc_belief":[],
        "p_true":   [],
        "raw_cam":  [],
        "pc_err":   [],
        "p_err":    [],
    }

    start = arm_pc.fk()
    print(f"  Start: ({start[0]:.3f}, {start[1]:.3f})   "
          f"Goal: ({goal[0]:.3f}, {goal[1]:.3f})   noise σ={noise}")
    print()
    print(f"{'Step':>5} | {'PC  true':>16} | {'Belief':>16} | "
          f"{'P   true':>16} | {'PC err':>8} | {'P  err':>8}")
    print("─" * 92)

    for t in range(steps):
        # ── PC-controlled arm ──────────────────────────────────────────────────
        y_cam  = arm_pc.camera_reading(noise)       # noisy pixel reading
        belief = estimator.update(y_cam)            # PC: refine belief
        v_pc   = pc_ctrl.command(belief)            # visual active inference
        arm_pc.apply_cartesian_velocity(v_pc)       # move arm

        # ── P-controlled arm (no filtering) ───────────────────────────────────
        y_raw  = arm_p.camera_reading(noise)
        v_p    = p_ctrl.command(y_raw)
        arm_p.apply_cartesian_velocity(v_p)

        # ── Record ────────────────────────────────────────────────────────────
        pc_true = arm_pc.fk()
        p_true  = arm_p.fk()
        hist["pc_true"].append(pc_true)
        hist["pc_belief"].append(belief)
        hist["p_true"].append(p_true)
        hist["raw_cam"].append(y_cam)
        hist["pc_err"].append(float(np.linalg.norm(pc_true - goal)))
        hist["p_err"].append(float(np.linalg.norm(p_true  - goal)))

        if t % 15 == 0 or t == steps - 1:
            print(f"{t:5d} | ({pc_true[0]:6.3f},{pc_true[1]:6.3f}) | "
                  f"({belief[0]:6.3f},{belief[1]:6.3f}) | "
                  f"({p_true[0]:6.3f},{p_true[1]:6.3f}) | "
                  f"{hist['pc_err'][-1]:8.4f} | {hist['p_err'][-1]:8.4f}")

    return hist, goal


# ── 6.  ASCII 2-D map ─────────────────────────────────────────────────────────

def ascii_map(hist, goal, width=56, height=20):
    """Top-down ASCII view of the 2-D marker trajectories."""
    pc_traj  = hist["pc_true"]
    p_traj   = hist["p_true"]
    bel_traj = hist["pc_belief"]
    start    = pc_traj[0]

    all_pts  = pc_traj + p_traj + [start, goal]
    all_x    = [p[0] for p in all_pts]
    all_y    = [p[1] for p in all_pts]
    xmin, xmax = min(all_x) - 0.1, max(all_x) + 0.1
    ymin, ymax = min(all_y) - 0.1, max(all_y) + 0.1
    xrng = xmax - xmin + 1e-9
    yrng = ymax - ymin + 1e-9

    canvas = [[" "] * width for _ in range(height)]

    def col(x): return int((x - xmin) / xrng * (width - 1))
    def row(y): return height - 1 - int((y - ymin) / yrng * (height - 1))

    def plot(traj, ch):
        for p in traj:
            c, r = col(p[0]), row(p[1])
            if 0 <= r < height and 0 <= c < width:
                if canvas[r][c] == " ":
                    canvas[r][c] = ch

    plot(bel_traj, "·")     # PC belief path
    plot(p_traj,   "o")     # P-controller path
    plot(pc_traj,  "*")     # PC true path

    # Mark start and goal (overwrite any trajectory char)
    sc, sr = col(start[0]), row(start[1])
    gc, gr = col(goal[0]),  row(goal[1])
    if 0 <= sr < height and 0 <= sc < width: canvas[sr][sc] = "S"
    if 0 <= gr < height and 0 <= gc < width: canvas[gr][gc] = "G"

    print("\n  2-D marker trajectory (top-down view)")
    print("  ┌" + "─" * width + "┐")
    for r in canvas:
        print("  │" + "".join(r) + "│")
    print("  └" + "─" * width + "┘")
    print()
    print("  S = start     G = goal")
    print("  * = PC arm (true)    · = PC belief    o = P-controller arm")


# ── 7.  Results summary ───────────────────────────────────────────────────────

def summary(hist, goal):
    T    = len(hist["pc_err"])
    half = T // 2

    raw_rmse = np.sqrt(np.mean(
        [np.linalg.norm(r - t) ** 2
         for r, t in zip(hist["raw_cam"], hist["pc_true"])]
    ))
    bel_rmse = np.sqrt(np.mean(
        [np.linalg.norm(b - t) ** 2
         for b, t in zip(hist["pc_belief"], hist["pc_true"])]
    ))

    print()
    print("=" * 58)
    print("  Results")
    print("=" * 58)
    print(f"  Final PC  distance to goal : {hist['pc_err'][-1]:.4f} m")
    print(f"  Final P   distance to goal : {hist['p_err'][-1]:.4f} m")
    print(f"  Mean PC  error  (t ≥ {half:3d}) : {np.mean(hist['pc_err'][half:]):.4f} m")
    print(f"  Mean P   error  (t ≥ {half:3d}) : {np.mean(hist['p_err'][half:]):.4f} m")
    print()
    print(f"  Raw camera RMSE vs. true   : {raw_rmse:.4f} m")
    print(f"  PC belief RMSE vs. true    : {bel_rmse:.4f} m")
    if raw_rmse > 0:
        print(f"  Noise reduction            : {100*(1 - bel_rmse/raw_rmse):.1f}%")
    print()
    print("  PC belief filters camera noise → smoother tracking and")
    print("  less motor jitter than the raw-camera P-controller.")


# ── 8.  Real hardware notes ───────────────────────────────────────────────────

def hardware_notes():
    print()
    print("=" * 65)
    print("  Connecting to Real Hardware")
    print("=" * 65)
    print("""
  ── CAMERA  (replace arm.camera_reading()) ─────────────────────────

    import cv2
    cap   = cv2.VideoCapture(0)
    _, frame = cap.read()
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, lower_red, upper_red)   # tune bounds
    M     = cv2.moments(mask)
    cx    = M['m10'] / (M['m00'] + 1e-9)   # pixel x
    cy    = M['m01'] / (M['m00'] + 1e-9)   # pixel y
    y_cam = np.array([cx, cy])             # → feed to estimator.update()

    Goal pixel coords: snap a photo of arm at target → read (cx, cy).

  ── MOTORS  (replace arm.apply_cartesian_velocity(v)) ──────────────

    Option A — Arduino / serial (receives vx, vy floats):
      ser.write(f"{v[0]:.3f},{v[1]:.3f}\\n".encode())
      # MCU firmware applies Jacobian-transpose on-board

    Option B — ROS2:
      twist.linear.x = v[0]; twist.linear.y = v[1]
      pub.publish(twist)

    Option C — Direct servo (Jacobian on microcontroller):
      # MCU receives (vx, vy), computes J_pinv @ v → dtheta, moves servos

  ── TUNING GUIDE ───────────────────────────────────────────────────
    Arm too slow or stops short   → increase kappa_goal  (1.0 – 3.0)
    Arm overshoots / oscillates   → decrease kappa_goal  or max_speed
    Belief lags behind real arm   → increase kappa_sensory (2.0 – 6.0)
    Belief jitters with noise     → decrease kappa_sensory / increase kappa_prior
    Arm jitters even at goal      → lower max_speed or add dead-band:
                                    if np.linalg.norm(v) < 0.01: v = 0
  """)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 92)
    print("  Visual Active Inference — 2-DOF Arm with Camera  (pc_visual_robot.py)")
    print("  PC state estimator + Active Inference controller — no inverse kinematics")
    print("=" * 92)
    print()

    hist, goal = run_simulation(goal_xy=(0.8, 1.2), steps=120, noise=PIXEL_NOISE)
    ascii_map(hist, goal)
    summary(hist, goal)
    hardware_notes()

    # ── Second run: high camera noise ─────────────────────────────────────────
    print()
    print("=" * 92)
    print("  High-noise environment (σ = 0.25 m) — PC belief absorbs camera noise")
    print("=" * 92)
    print()
    hist2, goal2 = run_simulation(goal_xy=(0.8, 1.2), steps=120, noise=0.25)
    ascii_map(hist2, goal2)
    summary(hist2, goal2)
