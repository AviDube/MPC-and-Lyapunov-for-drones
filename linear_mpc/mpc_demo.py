""""
Linear MPC demo on MuJoCo
"""

import mujoco
import mujoco.viewer
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Config
XML_PATH = "../basic_quadrotor.xml"
DT_CTRL = 0.02
SIM_TIME = 8.0

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

m = model.body_mass[1]
g = abs(model.opt.gravity[2])
Ix, Iy, Iz = model.body_inertia[1]

HOVER = (m * g) / 4

def quat_to_euler(q):
    w, x, y, z = q
    r = Rotation.from_quat([x, y, z, w])
    return r.as_euler("xyz")

def get_state(data):
    pos   = data.qpos[:3]
    quat  = data.qpos[3:7]
    euler = quat_to_euler(quat)
    vel   = data.qvel[:3]
    omega = data.qvel[3:6]
    return np.concatenate([pos, euler, vel, omega])

# linearized model
nx = 12
nu = 4  # [T, tau_x, tau_y, tau_z]

A = np.zeros((nx, nx))
B = np.zeros((nx, nu))

# position kinematics
A[0,6] = 1
A[1,7] = 1
A[2,8] = 1

# small-angle coupling
A[6,4] = g     
A[7,3] = -g 

# angular kinematics
A[3,9]  = 1
A[4,10] = 1
A[5,11] = 1

# control effects
B[8,0]  = 1/m
B[9,1]  = 1/Ix
B[10,2] = 1/Iy
B[11,3] = 1/Iz

# discretize
Ad = np.eye(nx) + A * DT_CTRL
Bd = B * DT_CTRL


def wrench_to_rotors(u):
    T, tx, ty, tz = u
    l = 0.028    # arm length from XML site positions
    k = 0.02513  # yaw coefficient magnitude from XML gear

    u0 = T/4 - tx/(4*l) - ty/(4*l) - tz/(4*k)
    u1 = T/4 - tx/(4*l) + ty/(4*l) + tz/(4*k)
    u2 = T/4 + tx/(4*l) + ty/(4*l) - tz/(4*k)
    u3 = T/4 + tx/(4*l) - ty/(4*l) + tz/(4*k)

    return np.array([u0, u1, u2, u3])

# mpc
class MPC:
    def __init__(self, horizon=50):
        self.N = horizon

        # State cost
        self.Q = np.diag([
            20, 20, 100,   # position
            5, 5, 50,      # roll, pitch, yaw
            2, 2, 5,       # linear velocities
            1, 1, 10       # angular velocities
        ])
        self.Qf = self.Q * 10

        # Control cost
        self.R = 0.01 * np.eye(nu)

        # Rate penalty
        self.Rdu = 0.1 * np.eye(nu)

        # Integral action weights [x, y, z, yaw]
        self.Ki = np.array([5.0, 5.0, 20.0, 5.0])

        self.integral_error = np.zeros(4)
        self.integral_clip  = np.array([0.3, 0.3, 1.5, 0.3])  # anti-windup

        # Track previous solution for rate penalty
        self.u_prev = np.array([m*g, 0.0, 0.0, 0.0])

    def solve(self, x0, x_ref):
        # Accumulate integral error on [x, y, z, yaw]
        err = x_ref[[0,1,2,5]] - x0[[0,1,2,5]]
        self.integral_error += err * DT_CTRL

        # Anti-windup clamp
        self.integral_error = np.clip(
            self.integral_error, -self.integral_clip, self.integral_clip
        )

        # Tilt compensation
        roll  = x0[3]
        pitch = x0[4]
        cos_tilt = np.cos(roll) * np.cos(pitch)
        cos_tilt = np.clip(cos_tilt, 0.5, 1.0)  # prevent divide by 0
        tilt_thrust = m * g / cos_tilt           # thrust needed to hold altitude

        # Feedforward wrench: tilt-compensated gravity + integral corrections
        u_ff = np.array([
            (tilt_thrust - m*g) + self.Ki[2] * self.integral_error[2],  # z
            -self.Ki[1] * self.integral_error[1],  # y error = roll torque
             self.Ki[0] * self.integral_error[0],  # x error = pitch torque
            self.Ki[3] * self.integral_error[3],   # yaw error = yaw torque
        ])
        u_hover = np.array([m*g, 0.0, 0.0, 0.0]) + u_ff

        x = cp.Variable((nx, self.N+1))
        u = cp.Variable((nu, self.N))

        cost = 0
        constraints = [x[:,0] == x0]
        c = np.zeros(nx)
        c[8] = -g * DT_CTRL

        for k in range(self.N):
            cost += cp.quad_form(x[:,k] - x_ref, self.Q)
            cost += cp.quad_form(u[:,k] - u_hover, self.R)

            # Rate penalty: penalize large changes in control input
            if k == 0:
                cost += cp.quad_form(u[:,k] - self.u_prev, self.Rdu)
            else:
                cost += cp.quad_form(u[:,k] - u[:,k-1], self.Rdu)

            constraints += [
                x[:,k+1] == Ad @ x[:,k] + Bd @ u[:,k] + c,
                u[0,k] >= 0.0,
                u[0,k] <= 2*m*g,
                cp.abs(u[1,k]) <= 0.005,
                cp.abs(u[2,k]) <= 0.005,
                cp.abs(u[3,k]) <= 0.02,
            ]

        cost += cp.quad_form(x[:,self.N] - x_ref, self.Qf)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if u.value is None:
            print("MPC infeasible, using previous solution")
            return self.u_prev

        self.u_prev = u[:,0].value
        return u[:,0].value

mpc = MPC()

# target hover
x_ref = np.zeros(nx)
x_ref[0] = 0.5
x_ref[1] = -0.5
x_ref[2] = 1.0
x_ref[5] = 0.0  # desired yaw

# diagnostics - remove when not needed
test_u = wrench_to_rotors(np.array([m*g, 0, 0, 0]))
print(f"m={m:.4f} kg, g={g:.4f} m/s^2, m*g={m*g:.4f} N")
print(f"Hover rotor commands: {test_u}")
print(f"Rotor sum: {sum(test_u):.4f} N  (should equal m*g={m*g:.4f} N)")
print(f"HOVER constant: {HOVER:.4f} N per rotor")


states   = []
controls = []
times    = []


# Check actuator gear and control range
print("Actuator gear vectors:")
for i in range(model.nu):
    print(f"  rotor{i}: gear={model.actuator_gear[i]}")

print(f"\nControl range: {model.actuator_ctrlrange}")
print(f"Force range:   {model.actuator_forcerange}")

print(f"\nActuator biasprm: {model.actuator_biasprm}")
print(f"Actuator gainprm: {model.actuator_gainprm}")


with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running() and t < SIM_TIME:
        x = get_state(data)

        # MPC gives wrench [T, tau_x, tau_y, tau_z]
        u_wrench = mpc.solve(x, x_ref)

        if len(times) % 25 == 0:  # print every 0.5s
            print(f"t={t:.1f} | z={x[2]:.3f} | z_err={x_ref[2]-x[2]:.3f} | "
            f"T_cmd={u_wrench[0]:.4f} | int_z={mpc.integral_error[2]:.4f} | "
            f"rotors={wrench_to_rotors(u_wrench)}")

        # Convert to individual rotor thrusts
        u_rotors = wrench_to_rotors(u_wrench)
        u_rotors = np.clip(u_rotors, 0.005, 0.25)

        data.ctrl[:] = u_rotors

        states.append(x.copy())
        controls.append(u_rotors.copy())
        times.append(t)

        for _ in range(int(DT_CTRL / model.opt.timestep)):
            mujoco.mj_step(model, data)

        viewer.sync()
        t += DT_CTRL

states   = np.array(states)
controls = np.array(controls)
times    = np.array(times)

plt.figure()
plt.plot(times, states[:,0], label="x")
plt.plot(times, states[:,1], label="y")
plt.plot(times, states[:,2], label="z")
plt.axhline(x_ref[0], linestyle="--", color="r", alpha=0.6, label="x_ref")
plt.axhline(x_ref[1], linestyle="--", color="g", alpha=0.6, label="y_ref")
plt.axhline(x_ref[2], linestyle="--", color="b", alpha=0.6, label="z_ref")
plt.title("Position Tracking")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()
plt.grid()

plt.figure()
plt.plot(times, states[:,3], label="roll")
plt.plot(times, states[:,4], label="pitch")
plt.plot(times, states[:,5], label="yaw")
plt.axhline(x_ref[5], linestyle="--", color="k", alpha=0.6, label="yaw_ref")
plt.title("Orientation")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid()

plt.figure()
for i in range(4):
    plt.plot(times, controls[:,i], label=f"rotor {i}", alpha=0.7)
plt.axhline(HOVER, linestyle="--", color="k", alpha=0.4, label="hover/4")
plt.title("Rotor Thrusts")
plt.xlabel("Time (s)")
plt.ylabel("Thrust (N)")
plt.legend()
plt.grid()

plt.show()