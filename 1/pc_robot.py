"""
pc_robot.py — Predictive Coding applied to a simulated 1-D robot.

Demonstrates three PC concepts in one place:

  1. PERCEPTION  — PC as a Bayesian state estimator.
                   A generative model predicts sensor readings;
                   prediction errors update the belief about position.
                   Produces a smooth estimate despite noisy sensors.

  2. CONTROL     — Active Inference motor commands.
                   The robot has a preferred (goal) state as its prior.
                   The motor command is proportional to the prediction
                   error between the desired and believed position —
                   i.e. "act to make sensory predictions come true."

  3. REFLEX ARC  — Same PC principle, applied at a lower level:
                   proprioceptive prediction error drives joint torque.
                   Biologically, this is how spinal reflexes work.

The robot lives on a 1-D track.  At each time step:
  a) It gets a noisy position reading from a sensor.
  b) Its PC belief (state estimate) is updated via prediction error.
  c) A motor command is generated from the goal prediction error.
  d) The command moves the robot (with simulated dynamics).

Run:  python3 pc_robot.py
"""

import numpy as np

np.random.seed(0)


# ─────────────────────────────────────────────────────────────────────────────
#  1. SIMULATED ROBOT (ground truth physics)
# ─────────────────────────────────────────────────────────────────────────────

class Robot1D:
    """
    Simple 1-D point robot on a track.
    State: [position, velocity]
    Dynamics: Euler integration with drag.
    Sensor: position + Gaussian noise.
    """

    def __init__(self, pos=0.0, sensor_noise=0.5, mass=1.0, drag=0.4):
        self.pos   = pos
        self.vel   = 0.0
        self.noise = sensor_noise
        self.mass  = mass
        self.drag  = drag           # velocity decay per step

    def step(self, force, dt=0.1):
        """Apply force, integrate dynamics."""
        acc      = force / self.mass
        self.vel = self.vel * (1 - self.drag) + acc * dt
        self.pos = self.pos + self.vel * dt

    def sense(self):
        """Return noisy position reading."""
        return self.pos + np.random.randn() * self.noise


# ─────────────────────────────────────────────────────────────────────────────
#  2. PC PERCEPTION — state estimator (generative model over position)
# ─────────────────────────────────────────────────────────────────────────────

class PCStateEstimator:
    """
    Minimalist predictive coding state estimator.

    Generative model:
        hidden state  :  μ  (believed position)
        prediction    :  ŷ = μ           (we expect to sense our believed position)
        sensory error :  ε_s = y - ŷ    (actual reading minus prediction)
        prior error   :  ε_p = μ - μ₀   (deviation from prior / last estimate)

    Belief update (gradient descent on variational free energy F):
        dμ/dt = κ_s · ε_s - κ_p · ε_p

    where κ_s and κ_p are precisions (inverse variances) that weight
    how much to trust sensors vs. prior continuity.
    """

    def __init__(self, init_pos=0.0,
                 kappa_sensory=2.0,   # trust sensors
                 kappa_prior=0.5,     # trust continuity of motion
                 steps=20):
        self.mu    = init_pos          # current belief (position)
        self.mu0   = init_pos          # prior (previous belief, carried forward)
        self.k_s   = kappa_sensory
        self.k_p   = kappa_prior
        self.steps = steps             # inference iterations per time step

    def update(self, y_sensed, predicted_pos=None):
        """
        Update belief given a new sensor reading y_sensed.
        predicted_pos: where dynamics say we should be (optional prior).
        """
        if predicted_pos is not None:
            self.mu0 = predicted_pos   # use dynamics prediction as prior

        mu = self.mu
        for _ in range(self.steps):
            eps_sensory = y_sensed - mu            # sensor prediction error
            eps_prior   = mu - self.mu0            # prior prediction error
            dmu = self.k_s * eps_sensory - self.k_p * eps_prior
            mu  = mu + 0.05 * dmu                  # small gradient step

        self.mu = mu
        return self.mu


# ─────────────────────────────────────────────────────────────────────────────
#  3. ACTIVE INFERENCE CONTROLLER (PC-based motor commands)
# ─────────────────────────────────────────────────────────────────────────────

class ActiveInferenceController:
    """
    Motor commands via active inference.

    The agent has a PREFERRED state (goal position) encoded as a prior.
    The motor command is the gradient that minimises the prediction error
    between the preferred and believed position:

        u = κ_goal · (goal - μ)

    This is a "reflex arc" derived from the free-energy principle:
    act to make sensory predictions come true, not to correct errors
    by updating beliefs alone.

    In biological terms:
      - goal encodes desired proprioceptive signal (efference copy)
      - motor neurons fire to minimise proprioceptive prediction error
      - spinal cord reflexes implement exactly this loop
    """

    def __init__(self, goal, kappa_goal=3.0, max_force=5.0):
        self.goal       = goal
        self.k_goal     = kappa_goal
        self.max_force  = max_force

    def command(self, believed_pos):
        """Return force to apply to robot."""
        eps_goal = self.goal - believed_pos         # goal prediction error
        u = self.k_goal * eps_goal
        return float(np.clip(u, -self.max_force, self.max_force))

    def set_goal(self, new_goal):
        self.goal = new_goal


# ─────────────────────────────────────────────────────────────────────────────
#  4. SIMPLE PID CONTROLLER (classical baseline for comparison)
# ─────────────────────────────────────────────────────────────────────────────

class PIDController:
    def __init__(self, goal, kp=3.0, ki=0.1, kd=0.5, max_force=5.0):
        self.goal  = goal
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max   = max_force
        self._int  = 0.0
        self._prev = 0.0

    def command(self, measured_pos):
        err       = self.goal - measured_pos
        self._int = np.clip(self._int + err * 0.1, -5, 5)
        deriv     = err - self._prev
        self._prev = err
        u = self.kp * err + self.ki * self._int + self.kd * deriv
        return float(np.clip(u, -self.max, self.max))


# ─────────────────────────────────────────────────────────────────────────────
#  5. SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(goal=5.0, steps=80, sensor_noise=0.5):
    """
    Run both PC/Active Inference and PID robots side by side.
    Robot starts at position 0, goal is at `goal`.
    """

    # Two independent robots with the same physics
    robot_pc  = Robot1D(pos=0.0, sensor_noise=sensor_noise)
    robot_pid = Robot1D(pos=0.0, sensor_noise=sensor_noise)

    estimator  = PCStateEstimator(init_pos=0.0, kappa_sensory=2.0, kappa_prior=0.5)
    controller = ActiveInferenceController(goal=goal, kappa_goal=3.0)
    pid        = PIDController(goal=goal, kp=3.0, ki=0.1, kd=0.5)

    print(f"Goal position: {goal:.1f}   Start: 0.0   Sensor noise σ={sensor_noise}")
    print()
    print(f"{'Step':>5} | {'True (PC)':>10} | {'Belief':>8} | {'Sensor':>8} | "
          f"{'Force':>7} | {'True (PID)':>11} | {'Err PC':>8} | {'Err PID':>8}")
    print("-" * 90)

    history = {"pc_true": [], "pc_belief": [], "pid_true": [],
               "sensor": [], "pc_err": [], "pid_err": []}

    for t in range(steps):
        # ── PC robot ──────────────────────────────────────────────────────────
        y = robot_pc.sense()                                  # noisy sensor
        # Prior: predict where we'll be based on last belief + expected motion
        predicted = estimator.mu + 0.0                        # stationary prior
        belief = estimator.update(y, predicted_pos=predicted) # update belief
        force_pc  = controller.command(belief)                # active inference
        robot_pc.step(force_pc)

        # ── PID robot (uses raw noisy sensor directly) ────────────────────────
        y_pid     = robot_pid.sense()
        force_pid = pid.command(y_pid)
        robot_pid.step(force_pid)

        # ── Record ────────────────────────────────────────────────────────────
        history["pc_true"].append(robot_pc.pos)
        history["pc_belief"].append(belief)
        history["pid_true"].append(robot_pid.pos)
        history["sensor"].append(y)
        history["pc_err"].append(abs(robot_pc.pos - goal))
        history["pid_err"].append(abs(robot_pid.pos - goal))

        if t % 8 == 0 or t == steps - 1:
            print(f"{t:5d} | {robot_pc.pos:10.3f} | {belief:8.3f} | {y:8.3f} | "
                  f"{force_pc:7.3f} | {robot_pid.pos:11.3f} | "
                  f"{history['pc_err'][-1]:8.3f} | {history['pid_err'][-1]:8.3f}")

    return history


def ascii_trajectory(history, goal, width=70, height=14):
    """Show both robots' trajectories as ASCII art."""
    pc   = history["pc_true"]
    pid  = history["pid_true"]
    bel  = history["pc_belief"]
    T    = len(pc)

    all_vals = pc + pid + bel + [goal, 0.0]
    ymin, ymax = min(all_vals) - 0.3, max(all_vals) + 0.3
    yrange = ymax - ymin + 1e-9

    canvas = [[" "] * width for _ in range(height)]

    def plot(series, char):
        for xi, val in enumerate(series):
            col = int((xi / (T - 1)) * (width - 1))
            row = height - 1 - int(((val - ymin) / yrange) * (height - 1))
            row = max(0, min(height - 1, row))
            canvas[row][col] = char

    # goal line
    goal_row = height - 1 - int(((goal - ymin) / yrange) * (height - 1))
    goal_row = max(0, min(height - 1, goal_row))
    for col in range(width):
        if canvas[goal_row][col] == " ":
            canvas[goal_row][col] = "-"

    plot(bel,  "·")   # PC belief
    plot(pc,   "*")   # PC true position
    plot(pid,  "o")   # PID true position

    print("\n  Position vs. time")
    for r, row in enumerate(canvas):
        if r % (height // 4) == 0:
            val = ymax - (r / (height - 1)) * yrange
            prefix = f"{val:6.2f} |"
        else:
            prefix = "       |"
        print(prefix + "".join(row))
    print("       +" + "-" * width)
    print(f"       t=0{' ' * (width - 10)}t={T}")
    print()
    print("  *  PC robot (true position)    ·  PC belief (state estimate)")
    print("  o  PID robot (true position)   --- goal")


def summary(history, goal):
    pc_err  = history["pc_err"]
    pid_err = history["pid_err"]
    T       = len(pc_err)
    settle  = max(T // 2, 1)

    print()
    print("=" * 50)
    print("  Results")
    print("=" * 50)
    print(f"  Final PC  position error : {pc_err[-1]:.4f}")
    print(f"  Final PID position error : {pid_err[-1]:.4f}")
    print(f"  Mean PC  error (t≥{settle}): {np.mean(pc_err[settle:]):.4f}")
    print(f"  Mean PID error (t≥{settle}): {np.mean(pid_err[settle:]):.4f}")
    sensor_rmse = np.sqrt(np.mean(
        [(s - t) ** 2 for s, t in zip(history["sensor"], history["pc_true"])]
    ))
    belief_rmse = np.sqrt(np.mean(
        [(b - t) ** 2 for b, t in zip(history["pc_belief"], history["pc_true"])]
    ))
    print(f"\n  Raw sensor RMSE vs. true pos : {sensor_rmse:.4f}")
    print(f"  PC belief RMSE vs. true pos  : {belief_rmse:.4f}")
    print(f"  Noise reduction: {100*(1 - belief_rmse/sensor_rmse):.1f}%")
    print()
    print("  The PC belief is smoother than the raw sensor because")
    print("  prediction errors are precision-weighted — the same mechanism")
    print("  the brain uses to filter noisy proprioception.")


def what_to_do_next():
    print()
    print("=" * 60)
    print("  How to extend this to a real robot")
    print("=" * 60)
    print("""
  1. REPLACE Robot1D.sense() with your actual sensor driver.
     e.g. for a servo:  read_encoder() → noisy angle reading

  2. REPLACE Robot1D.step() with your actuator driver.
     e.g. for a servo:  set_pwm(force)  or  send_torque(force)

  3. EXTEND the state to [position, velocity, acceleration]:
       estimator.mu = np.array([pos, vel, acc])
       Use a transition matrix A so prior = A @ mu (momentum)

  4. FOR MULTI-JOINT robots, stack joints in the state vector
     and use a Jacobian to map joint errors → Cartesian goal error.

  5. FOR NAVIGATION, the 'goal' becomes a waypoint from a planner.
     Switch goals dynamically with controller.set_goal(next_wp).

  6. FOR OBSTACLE AVOIDANCE, add a repulsive prediction error:
       eps_obstacle = 1/(distance_to_obstacle + ε)
       u += kappa_obs * eps_obstacle  (push away from obstacle)

  7. TO ADD LEARNING, connect pc_numpy.py's PCNetwork as the
     generative model A inside the state estimator so the robot
     learns its own sensor model from experience.
  """)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 90)
    print("  Predictive Coding Robot Controller — 1D Track Simulation")
    print("=" * 90)
    print()

    history = run_simulation(goal=5.0, steps=80, sensor_noise=0.5)
    ascii_trajectory(history, goal=5.0)
    summary(history, goal=5.0)
    what_to_do_next()

    # ── Second run: harder noise ──────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  High-noise environment (σ = 1.5) — PC belief filters more aggressively")
    print("=" * 90)
    print()
    history2 = run_simulation(goal=5.0, steps=80, sensor_noise=1.5)
    summary(history2, goal=5.0)
