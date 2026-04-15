# Predictive Coding in Hardware

Can the PC model be wired into a circuit? Yes — at several different levels,
and it maps surprisingly naturally to hardware.

---

## Three Ways to Wire It In

---

### 1. Digital — plug in sensors and motors (easiest)

`pc_robot.py` already runs on a Raspberry Pi. Replace two methods:

```python
def sense(self):
    return read_encoder()     # GPIO pulse counter, I2C IMU, lidar, etc.

def step(self, force, dt=0.1):
    set_pwm(force)            # RPi.GPIO PWM → motor driver → motor
```

The rest of the PC code runs unchanged in Python at ~1 kHz on a Pi Zero,
fast enough for most servo and motor control loops.

---

### 2. Microcontroller — port the estimator to C (no OS needed)

`PCStateEstimator` is a few lines of arithmetic. It fits on an Arduino:

```c
// PC state estimator — runs on any microcontroller
float mu = 0.0, mu0 = 0.0;
float kS = 2.0, kP = 0.5;

float pc_update(float sensor) {
    for (int i = 0; i < 20; i++) {
        float eps_s = sensor - mu;   // sensory prediction error
        float eps_p = mu - mu0;      // prior prediction error
        mu += 0.05 * (kS * eps_s - kP * eps_p);
    }
    mu0 = mu;
    return mu;                       // smoothed belief
}

float pc_control(float belief, float goal) {
    return 3.0 * (goal - belief);    // active inference motor command
}
```

Compiles to ~200 bytes. Runs at 100 kHz+ on an ATmega328 (Arduino Uno).

---

### 3. Analog circuit — the equations ARE circuits

This is where it gets genuinely interesting. The PC activity update is a
continuous-time differential equation:

```
dx/dt  =  −ε_sensory  +  W · ε_prior
```

That is an RC integrator with two input currents — directly implementable
with an op-amp:

```
Sensor ──[R_s]──┐
                ├──[op-amp integrator C]── belief voltage μ
Prior  ──[R_p]──┘              |
                                └── feeds back to compute ε
```

#### Component mapping

| PC concept | Circuit element |
|---|---|
| Neuron state x | Voltage on capacitor |
| dx/dt = ... | Op-amp integrator (RC) |
| Prediction ŷ = W · x | Resistor divider or multiplying DAC |
| Error ε = x − ŷ | Differential amplifier (op-amp subtractor) |
| ReLU nonlinearity | Diode clamp |
| Hebbian weight update ΔW = ε · x | Multiplier IC (e.g. AD633) or memristor |

#### Minimal 2-layer analog PC circuit

A sensor fusion circuit that implements one layer of PC needs roughly:

- 4 op-amps — TL074 quad package (~£1)
- 2 capacitors — neuron state storage
- 6 resistors — fixed weights and feedback paths
- 2 diodes — ReLU clamp

#### The deeper point

Real neurons are literally RC circuits. A neuron's membrane is a capacitor;
ion channels are resistors; the membrane potential obeys:

```
C · dV/dt  =  −g_leak · (V − E_leak)  +  I_synaptic
```

This has the same form as the PC activity update. The brain may actually
implement these equations in hardware. The Python simulation is a model of
what the analog circuit already does.

---

## What Is Realistic to Build

| Goal | Route | Effort |
|---|---|---|
| Mobile robot / rover controller | Raspberry Pi + `pc_robot.py` | Low |
| Servo with onboard state filtering | Arduino + C port above | Medium |
| Analog sensor fusion demo on breadboard | Op-amp PC circuit | Medium |
| Learnable weights in hardware | Above + AD633 multipliers or memristors | High |
| Large networks on neuromorphic chip | Intel Loihi / SpiNNaker | Research level |

---

## Recommended Path for a First Build

```
1. Raspberry Pi Zero W (~£15)
2. Any sensor with I2C or UART output (IMU, ultrasonic, encoder)
3. Any PWM motor driver (L298N, DRV8833)
4. Run pc_robot.py with sense() and step() replaced
5. Tune kappa_sensory and kappa_goal until the robot behaves well
6. Optionally move PCStateEstimator to a dedicated Arduino co-processor
   so the Pi handles planning while the Arduino runs the tight control loop
```

---

## Further Reading

- **Analog VLSI implementation of PC:** Deneve et al. (2001) — neural circuits
  for Bayesian inference, directly maps to op-amp designs
- **Memristive Hebbian learning:** Strukov et al. (2008) — memristors as
  physical synapses that implement ΔW ∝ ε · x in hardware
- **Neuromorphic PC:** Furber et al. — SpiNNaker as a platform for running
  hierarchical predictive coding at scale
- **Spinal reflex as active inference:** Friston et al. (2010) — the gamma
  motor neuron system as an analog implementation of the controller in
  `pc_robot.py`
