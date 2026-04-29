"""
Benchmarking script for the linear + residual MPC controller.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from mpc_residual import MPC, get_state, wrench_to_rotors
from benchmark_plots import plot_all_trajectories

XML_PATH = "../basic_quadrotor.xml"
DT_CTRL  = 0.02
nx       = 12

mj_model = mujoco.MjModel.from_xml_path(XML_PATH)

def reset_sim(d):
    mujoco.mj_resetData(mj_model, d)
    mujoco.mj_forward(mj_model, d)


def make_trajectory(name, duration):
    T = duration
    if name == "hover":
        def ref(t):
            x=np.zeros(nx); x[:3]=[0.5,-0.5,1.0]; return x

    elif name == "slow_sweep":
        def ref(t):
            x=np.zeros(nx)
            x[0]=0.5*np.sin(2*np.pi*t/T); x[1]=0.5*np.cos(2*np.pi*t/T)-0.5
            x[2]=1.0; return x

    elif name == "fast_step":
        wpts=[[0,0,1],[1.5,0,1],[1.5,1.5,1],[0,1.5,1.5],[0,0,1]]
        def ref(t):
            x=np.zeros(nx); x[:3]=wpts[min(int(t/2),len(wpts)-1)]; return x

    elif name == "figure8":
        a=0.8
        def ref(t):
            th=2*np.pi*t/T; x=np.zeros(nx)
            x[0]=a*np.sin(th); x[1]=a*np.sin(th)*np.cos(th); x[2]=1.0
            x[6]=a*np.cos(th)*(2*np.pi/T)
            x[7]=a*(np.cos(th)**2-np.sin(th)**2)*(2*np.pi/T); return x

    elif name == "yaw_sweep":
        def ref(t):
            x=np.zeros(nx); x[:3]=[0.3,-0.3,1.0]
            x[5]=np.pi*np.sin(np.pi*t/T); return x

    elif name == "ood_spiral":
        def ref(t):
            x=np.zeros(nx); r=0.6; w=2*np.pi/(T/2)
            x[0]=r*np.cos(w*t); x[1]=r*np.sin(w*t)
            x[2]=0.5+0.8*(t/T); return x

    return ref


def compute_metrics(states, refs, times, solve_times):
    states=np.array(states); refs=np.array(refs)
    pos_err=states[:,:3]-refs[:,:3]; att_err=states[:,3:6]-refs[:,3:6]
    pos_rmse=float(np.sqrt((pos_err**2).mean()))
    att_rmse=float(np.sqrt((att_err**2).mean()))
    peak_err=float(np.abs(pos_err).max())

    err_mag=np.linalg.norm(pos_err,axis=1)
    window=int(0.5/DT_CTRL); settling=float("inf")
    for i in range(len(err_mag)-window):
        if err_mag[i:i+window].max()<0.05:
            settling=times[i]; break

    return {
        "pos_rmse":      pos_rmse,
        "att_rmse":      att_rmse,
        "peak_err":      peak_err,
        "settling_s":    settling,
        "solve_ms_mean": float(np.mean(solve_times)*1e3),
        "solve_ms_p95":  float(np.percentile(solve_times,95)*1e3),
    }


def run_episode(traj_name, duration):
    mj_data=mujoco.MjData(mj_model); reset_sim(mj_data)
    mpc=MPC(); ref_traj=make_trajectory(traj_name,duration)
    times=np.arange(0,duration,DT_CTRL)
    states,refs,solve_times=[],[],[]

    for t in times:
        x=get_state(mj_data); x_ref=ref_traj(t)
        t0=time.time(); u=mpc.solve(x,x_ref)
        solve_times.append(time.time()-t0)
        u_rot=np.clip(wrench_to_rotors(u),0.005,0.25)
        mj_data.ctrl[:]=u_rot
        for _ in range(int(DT_CTRL/mj_model.opt.timestep)):
            mujoco.mj_step(mj_model,mj_data)
        states.append(x); refs.append(x_ref)

    metrics=compute_metrics(states,refs,times,solve_times)
    return {"times":times.tolist(),"states":states,"refs":refs,
            "solve_times":solve_times,"metrics":metrics}


TRAJS    =["hover","slow_sweep","fast_step","figure8","yaw_sweep","ood_spiral"]
DURATIONS={"hover":6,"slow_sweep":8,"fast_step":10,
           "figure8":10,"yaw_sweep":8,"ood_spiral":8}

if __name__=="__main__":
    all_results={}
    for traj in TRAJS:
        print(f"Running {traj}...")
        r=run_episode(traj,DURATIONS[traj])
        all_results[traj]=r
        m=r["metrics"]
        print(f"  pos_rmse={m['pos_rmse']:.4f}m  "
              f"settle={m['settling_s']:.2f}s  "
              f"solve={m['solve_ms_mean']:.1f}ms")

    plot_all_trajectories(all_results,
                          controller_label="Linear + Residual MPC",
                          save_prefix="benchmark_residual",
                          color="#EF9F27")