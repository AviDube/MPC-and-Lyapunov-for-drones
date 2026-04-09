"""
physics.py
──────────
Exact nonlinear quadrotor physics model with RK4 integration.

This is the 'known' part of the hybrid model. It implements:
  - Full rotation matrix (no small-angle approximation)
  - Exact Euler angle kinematics (W matrix)
  - Gyroscopic coupling: omega x I omega
  - Gravity in world frame
  - Thrust mapped through full rotation matrix

Both numpy (for training) and CasADi (for MPC) implementations
are provided — they are mathematically identical.

State vector x = [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r]
Control u     = [T, tau_x, tau_y, tau_z]   (total thrust + body torques)
"""

import numpy as np
import casadi as ca


# ══════════════════════════════════════════════════════════════════════════════
# Numpy implementation (used during training data generation + loss computation)
# ══════════════════════════════════════════════════════════════════════════════

def rotation_matrix_np(phi, theta, psi):
    """ZYX rotation matrix R: body → world."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [-st,    ct*sp,             ct*cp            ],
    ])


def euler_kinematics_np(phi, theta):
    """
    W matrix: eta_dot = W(eta) @ omega
    Maps body angular rates (p,q,r) to Euler angle rates (phi_dot, theta_dot, psi_dot).
    Singular at theta = ±pi/2 but fine for typical drone flight.
    """
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, tt = np.cos(theta), np.tan(theta)
    return np.array([
        [1,  sp*tt,  cp*tt],
        [0,  cp,    -sp   ],
        [0,  sp/ct,  cp/ct],
    ])


def ode_np(x, u, mass, inertia, grav):
    """
    Continuous-time quadrotor ODE: xdot = f_physics(x, u)

    Parameters
    ----------
    x       : (12,) state vector
    u       : (4,)  control [T, tau_x, tau_y, tau_z]
    mass    : float
    inertia : (3,)  [Ix, Iy, Iz]
    grav    : float

    Returns
    -------
    xdot : (12,)
    """
    px,py,pz      = x[0],  x[1],  x[2]
    phi,theta,psi = x[3],  x[4],  x[5]
    vx,vy,vz      = x[6],  x[7],  x[8]
    p, q, r       = x[9],  x[10], x[11]
    T, tx,ty,tz   = u[0],  u[1],  u[2],  u[3]
    Ix,Iy,Iz      = inertia

    # Position kinematics
    pos_dot = np.array([vx, vy, vz])

    # Velocity dynamics: R @ [0,0,T/m] - [0,0,g]
    R   = rotation_matrix_np(phi, theta, psi)
    acc = R @ np.array([0, 0, T/mass]) - np.array([0, 0, grav])

    # Euler angle kinematics
    W       = euler_kinematics_np(phi, theta)
    eta_dot = W @ np.array([p, q, r])

    # Angular rate dynamics (Euler equations): I*omega_dot = tau - omega x I*omega
    omega   = np.array([p, q, r])
    I_omega = np.array([Ix*p, Iy*q, Iz*r])
    gyro    = np.cross(omega, I_omega)           # omega x I*omega (gyroscopic)
    tau     = np.array([tx, ty, tz])
    omega_dot = np.array([(tau[0] - gyro[0]) / Ix,
                          (tau[1] - gyro[1]) / Iy,
                          (tau[2] - gyro[2]) / Iz])

    return np.concatenate([pos_dot, eta_dot, acc, omega_dot])


def rk4_np(x, u, dt, mass, inertia, grav):
    """
    RK4 integration of the physics ODE over one timestep dt.
    4th-order accurate — much better than Euler for the same dt.
    """
    k1 = ode_np(x,            u, mass, inertia, grav)
    k2 = ode_np(x + dt/2*k1,  u, mass, inertia, grav)
    k3 = ode_np(x + dt/2*k2,  u, mass, inertia, grav)
    k4 = ode_np(x + dt*k3,    u, mass, inertia, grav)
    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


# ══════════════════════════════════════════════════════════════════════════════
# CasADi implementation (used inside the MPC NLP)
# ══════════════════════════════════════════════════════════════════════════════

def rotation_matrix_ca(phi, theta, psi):
    """ZYX rotation matrix in CasADi MX."""
    cp, sp = ca.cos(phi),   ca.sin(phi)
    ct, st = ca.cos(theta), ca.sin(theta)
    cy, sy = ca.cos(psi),   ca.sin(psi)
    row0 = ca.horzcat(cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp)
    row1 = ca.horzcat(sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp)
    row2 = ca.horzcat(-st,    ct*sp,              ct*cp)
    return ca.vertcat(row0, row1, row2)


def euler_kinematics_ca(phi, theta):
    """W matrix in CasADi MX."""
    cp, sp = ca.cos(phi),   ca.sin(phi)
    ct, tt = ca.cos(theta), ca.tan(theta)
    row0 = ca.horzcat(1,  sp*tt,  cp*tt)
    row1 = ca.horzcat(0,  cp,    -sp)
    row2 = ca.horzcat(0,  sp/ct,  cp/ct)
    return ca.vertcat(row0, row1, row2)


def ode_ca(x, u, mass, inertia, grav):
    """
    Continuous-time quadrotor ODE in CasADi MX.
    x, u are CasADi column vectors (MX).
    """
    phi, theta, psi = x[3], x[4], x[5]
    vx, vy, vz      = x[6], x[7], x[8]
    p,  q,  r       = x[9], x[10], x[11]
    T, tx, ty, tz   = u[0], u[1],  u[2],  u[3]
    Ix, Iy, Iz      = inertia

    pos_dot = ca.vertcat(vx, vy, vz)

    R   = rotation_matrix_ca(phi, theta, psi)
    acc = R @ ca.vertcat(0, 0, T/mass) - ca.vertcat(0, 0, grav)

    W       = euler_kinematics_ca(phi, theta)
    eta_dot = W @ ca.vertcat(p, q, r)

    omega   = ca.vertcat(p, q, r)
    I_omega = ca.vertcat(Ix*p, Iy*q, Iz*r)
    gyro    = ca.cross(omega, I_omega)
    tau     = ca.vertcat(tx, ty, tz)
    omega_dot = ca.vertcat(
        (tau[0] - gyro[0]) / Ix,
        (tau[1] - gyro[1]) / Iy,
        (tau[2] - gyro[2]) / Iz,
    )

    return ca.vertcat(pos_dot, eta_dot, acc, omega_dot)


def rk4_ca(x, u, dt, mass, inertia, grav):
    """RK4 integration in CasADi MX."""
    k1 = ode_ca(x,              u, mass, inertia, grav)
    k2 = ode_ca(x + dt/2 * k1,  u, mass, inertia, grav)
    k3 = ode_ca(x + dt/2 * k2,  u, mass, inertia, grav)
    k4 = ode_ca(x + dt    * k3, u, mass, inertia, grav)
    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


def build_physics_fn(mass, inertia, grav, dt,
                     name="f_physics"):
    """
    Build a CasADi Function for the physics step.
    f_physics(x, u) → x_next_physics
    """
    x_sym = ca.MX.sym("x", 12)
    u_sym = ca.MX.sym("u", 4)
    xn    = rk4_ca(x_sym, u_sym, dt, mass, inertia, grav)
    return ca.Function(name, [x_sym, u_sym], [xn], ["x", "u"], ["x_next"])