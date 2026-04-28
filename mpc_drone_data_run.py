"""
This script collects MPC-generated transition data for a quadrotor in MuJoCo.
"""

import argparse
import os
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation

XML_PATH  = "basic_quadrotor.xml"
DT_CTRL   = 0.02
SIM_TIME  = 8.0

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

m  = model.body_mass[1]
g  = abs(model.opt.gravity[2])
Ix, Iy, Iz = model.body_inertia[1]

def quat_to_euler(q):
    w, x, y, z = q
    r = Rotation.from_quat([x, y, z, w])
    return r.as_euler("xyz")

def get_state(d):
    pos   = d.qpos[:3]
    quat  = d.qpos[3:7]
    euler = quat_to_euler(quat)
    vel   = d.qvel[:3]
    omega = d.qvel[3:6]
    return np.concatenate([pos, euler, vel, omega])

def wrench_to_rotors(u):
    T, tx, ty, tz = u
    l = 0.028
    k = 0.02513
    u0 = T/4 - tx/(4*l) - ty/(4*l) - tz/(4*k)
    u1 = T/4 - tx/(4*l) + ty/(4*l) + tz/(4*k)
    u2 = T/4 + tx/(4*l) + ty/(4*l) - tz/(4*k)
    u3 = T/4 + tx/(4*l) - ty/(4*l) + tz/(4*k)
    return np.array([u0, u1, u2, u3])

# linearized model
nx, nu = 12, 4

A = np.zeros((nx, nx))
B = np.zeros((nx, nu))
A[0,6]=1; A[1,7]=1; A[2,8]=1
A[6,4]=g; A[7,3]=-g
A[3,9]=1; A[4,10]=1; A[5,11]=1
B[8,0]=1/m; B[9,1]=1/Ix; B[10,2]=1/Iy; B[11,3]=1/Iz

Ad = np.eye(nx) + A * DT_CTRL
Bd = B * DT_CTRL


try:
    from linear_mpc.mpc_demo import MPC 
except ImportError:
    import cvxpy as cp

    class MPC:
        def __init__(self, horizon=50):
            self.N = horizon
            self.Q  = np.diag([20,20,100, 5,5,50, 2,2,5, 1,1,10])
            self.Qf = self.Q * 10
            self.R  = 0.01 * np.eye(nu)
            self.Rdu = 0.1 * np.eye(nu)
            self.Ki = np.array([5.0, 5.0, 20.0, 5.0])
            self.integral_error = np.zeros(4)
            self.integral_clip  = np.array([0.3, 0.3, 1.5, 0.3])
            self.u_prev = np.array([m*g, 0.0, 0.0, 0.0])

        def solve(self, x0, x_ref):
            err = x_ref[[0,1,2,5]] - x0[[0,1,2,5]]
            self.integral_error += err * DT_CTRL
            self.integral_error  = np.clip(self.integral_error,
                                           -self.integral_clip, self.integral_clip)
            roll, pitch = x0[3], x0[4]
            cos_tilt = np.clip(np.cos(roll)*np.cos(pitch), 0.5, 1.0)
            tilt_thrust = m * g / cos_tilt
            u_ff = np.array([
                (tilt_thrust - m*g) + self.Ki[2]*self.integral_error[2],
                -self.Ki[1]*self.integral_error[1],
                 self.Ki[0]*self.integral_error[0],
                 self.Ki[3]*self.integral_error[3],
            ])
            u_hover = np.array([m*g, 0.0, 0.0, 0.0]) + u_ff

            x = cp.Variable((nx, self.N+1))
            u = cp.Variable((nu, self.N))
            cost = 0
            constraints = [x[:,0] == x0]
            c = np.zeros(nx); c[8] = -g * DT_CTRL

            for k in range(self.N):
                cost += cp.quad_form(x[:,k] - x_ref, self.Q)
                cost += cp.quad_form(u[:,k] - u_hover, self.R)
                u_prev_k = self.u_prev if k == 0 else u[:,k-1]
                cost += cp.quad_form(u[:,k] - u_prev_k, self.Rdu)
                constraints += [
                    x[:,k+1] == Ad @ x[:,k] + Bd @ u[:,k] + c,
                    u[0,k] >= 0.0, u[0,k] <= 2*m*g,
                    cp.abs(u[1,k]) <= 0.005,
                    cp.abs(u[2,k]) <= 0.005,
                    cp.abs(u[3,k]) <= 0.02,
                ]
            cost += cp.quad_form(x[:,self.N] - x_ref, self.Qf)
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if u.value is None:
                return self.u_prev
            self.u_prev = u[:,0].value
            return u[:,0].value


# collection loop 
def collect_episode(x_ref, render=False):
    """Run one episode, return (X, U_wrench, X_next) arrays."""
    mujoco.mj_resetData(model, data)
    mpc = MPC()

    X_buf, U_buf, Xn_buf, Uw_buf = [], [], [], []

    steps = int(SIM_TIME / DT_CTRL)
    viewer_ctx = mujoco.viewer.launch_passive(model, data) if render else None

    for _ in range(steps):
        x_t = get_state(data)

        u_wrench = mpc.solve(x_t, x_ref)
        u_rotors = np.clip(wrench_to_rotors(u_wrench), 0.005, 0.25)
        data.ctrl[:] = u_rotors

        for _ in range(int(DT_CTRL / model.opt.timestep)):
            mujoco.mj_step(model, data)

        x_next = get_state(data)

        X_buf.append(x_t)
        U_buf.append(u_wrench)          # store wrench for residual computation
        Xn_buf.append(x_next)
        Uw_buf.append(u_rotors)

        if viewer_ctx is not None:
            viewer_ctx.sync()

    if viewer_ctx is not None:
        viewer_ctx.close()

    return (np.array(X_buf), np.array(U_buf),
            np.array(Xn_buf), np.array(Uw_buf))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",      default="data/transitions.npz")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes with randomised start poses")
    parser.add_argument("--render",   action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Reference hover targets
    refs = [
        np.array([0.5, -0.5, 1.0, 0, 0, 0.0, 0,0,0, 0,0,0]),
        np.array([0.0,  0.0, 1.5, 0, 0, 0.3, 0,0,0, 0,0,0]),
        np.array([1.0,  1.0, 0.8, 0, 0,-0.3, 0,0,0, 0,0,0]),
    ]

    all_X, all_U, all_Xn, all_Uw = [], [], [], []

    for ep in range(args.episodes):
        x_ref = refs[ep % len(refs)]
        print(f"\nEpisode {ep+1}/{args.episodes}  ref={x_ref[:3]}")
        X, U, Xn, Uw = collect_episode(x_ref, render=args.render)
        all_X.append(X);  all_U.append(U)
        all_Xn.append(Xn); all_Uw.append(Uw)
        print(f"  collected {len(X)} transitions")

    X_all  = np.concatenate(all_X,  axis=0)
    U_all  = np.concatenate(all_U,  axis=0)
    Xn_all = np.concatenate(all_Xn, axis=0)
    Uw_all = np.concatenate(all_Uw, axis=0)

    np.savez(args.out, X=X_all, U_wrench=U_all, X_next=Xn_all, U_rotors=Uw_all)
    print(f"\nSaved {len(X_all)} transitions → {args.out}")
    print(f"  X shape      : {X_all.shape}")
    print(f"  U_wrench shape: {U_all.shape}")
    print(f"  X_next shape : {Xn_all.shape}")


if __name__ == "__main__":
    main()