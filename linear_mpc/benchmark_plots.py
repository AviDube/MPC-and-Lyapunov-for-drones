"""
Benchmark plotting utilities for linear MPC demo.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

DT_CTRL = 0.02
COLORS  = {
    "x": "#378ADD", "y": "#1D9E75", "z": "#D85A30",
    "roll": "#7F77DD", "pitch": "#D4537E", "yaw": "#BA7517",
    "ref": "#888780", "err": "#E24B4A", "settle": "#0F6E56",
}


# Per-trajectory plot: 4-row figure
#   Row 1: x, y, z position vs reference
#   Row 2: roll, pitch, yaw vs reference
#   Row 3: position error magnitude + settling time marker
#   Row 4: solve time per step
def plot_trajectory(result, traj_name, controller_label, ax_array):
    """
    Fill a column of 4 axes for one trajectory.
    ax_array: list of 4 matplotlib Axes (top to bottom)
    """
    times  = np.array(result["times"])
    states = np.array(result["states"])
    refs   = np.array(result["refs"])
    st     = result["solve_times"]
    m      = result["metrics"]

    ax_pos, ax_att, ax_err, ax_solve = ax_array

    # Row 1
    for i, (lbl, col) in enumerate(zip(["x","y","z"],
                                       [COLORS["x"],COLORS["y"],COLORS["z"]])):
        ax_pos.plot(times, states[:,i], color=col, lw=1.4, label=f"{lbl}")
        ax_pos.plot(times, refs[:,i],   color=col, lw=1.0,
                    ls="--", alpha=0.5)
    ax_pos.set_ylabel("Position (m)", fontsize=8)
    ax_pos.legend(fontsize=7, ncol=3, loc="upper right")
    ax_pos.grid(alpha=0.25); ax_pos.set_xlim(times[0], times[-1])
    ax_pos.set_title(traj_name.replace("_"," "), fontsize=10, fontweight="500")

    # Row 2 
    for i, (lbl, col) in enumerate(zip(["roll","pitch","yaw"],
                                       [COLORS["roll"],COLORS["pitch"],
                                        COLORS["yaw"]])):
        ax_att.plot(times, np.degrees(states[:,3+i]),
                    color=col, lw=1.4, label=lbl)
        ax_att.plot(times, np.degrees(refs[:,3+i]),
                    color=col, lw=1.0, ls="--", alpha=0.5)
    ax_att.set_ylabel("Angle (deg)", fontsize=8)
    ax_att.legend(fontsize=7, ncol=3, loc="upper right")
    ax_att.grid(alpha=0.25); ax_att.set_xlim(times[0], times[-1])

    # row 3
    pos_err = np.linalg.norm(states[:,:3] - refs[:,:3], axis=1)
    ax_err.plot(times, pos_err*100, color=COLORS["err"], lw=1.4,
                label="pos error")
    ax_err.axhline(5.0, color=COLORS["settle"], lw=1.0, ls="--",
                   label="5 cm threshold")

    # Shade settled region
    settling = m["settling_s"]
    if settling < float("inf"):
        ax_err.axvspan(settling, times[-1], alpha=0.08,
                       color=COLORS["settle"])
        ax_err.axvline(settling, color=COLORS["settle"], lw=1.2, ls=":")
        ax_err.text(settling + 0.05, ax_err.get_ylim()[1]*0.85,
                    f"settled\n{settling:.1f}s",
                    fontsize=7, color=COLORS["settle"])
    else:
        ax_err.text(0.98, 0.88, "never settled",
                    transform=ax_err.transAxes,
                    ha="right", fontsize=7, color=COLORS["err"])

    ax_err.set_ylabel("Pos error (cm)", fontsize=8)
    ax_err.legend(fontsize=7, loc="upper right")
    ax_err.grid(alpha=0.25); ax_err.set_xlim(times[0], times[-1])
    ax_err.set_ylim(bottom=0)

    # Annotate RMSE
    ax_err.text(0.02, 0.88,
                f"RMSE {m['pos_rmse']*100:.1f} cm",
                transform=ax_err.transAxes,
                fontsize=7, color=COLORS["err"])

    # Row 4
    solve_ms = np.array(st) * 1e3
    ax_solve.plot(times, solve_ms, color="#888780", lw=0.8, alpha=0.7)
    ax_solve.axhline(DT_CTRL*1e3, color="#E24B4A", lw=1.0, ls="--",
                     label=f"budget {DT_CTRL*1e3:.0f}ms")
    ax_solve.axhline(np.mean(solve_ms), color="#378ADD", lw=1.0, ls=":",
                     label=f"mean {np.mean(solve_ms):.1f}ms")
    ax_solve.set_ylabel("Solve (ms)", fontsize=8)
    ax_solve.set_xlabel("Time (s)", fontsize=8)
    ax_solve.legend(fontsize=7, loc="upper right")
    ax_solve.grid(alpha=0.25); ax_solve.set_xlim(times[0], times[-1])
    ax_solve.set_ylim(bottom=0)



def plot_summary(all_results, controller_label, color):
    trajs   = list(all_results.keys())
    metrics = ["pos_rmse","att_rmse","peak_err","settling_s","solve_ms_mean"]
    ylabels = ["Pos RMSE (m)","Att RMSE (rad)","Peak error (m)",
               "Settling time (s)","Mean solve (ms)"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.5))
    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        vals = []
        for t in trajs:
            v = all_results[t]["metrics"].get(metric, float("nan"))
            vals.append(v if v != float("inf") else float("nan"))

        bars = ax.bar(trajs, vals, color=color, alpha=0.85, width=0.55)
        ax.set_title(ylabel, fontsize=9)
        ax.set_xticklabels([t.replace("_","\n") for t in trajs],
                           fontsize=7)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()*1.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    fig.suptitle(f"{controller_label} — summary", fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


# orchestrator
def plot_all_trajectories(all_results, controller_label, save_prefix,
                          color="#378ADD"):
    trajs  = list(all_results.keys())
    n_cols = len(trajs)
    n_rows = 4   # pos, att, error, solve

    fig = plt.figure(figsize=(4.5*n_cols, 3.2*n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.45, wspace=0.35)

    for col, traj in enumerate(trajs):
        axes = [fig.add_subplot(gs[row, col]) for row in range(n_rows)]
        plot_trajectory(all_results[traj], traj, controller_label, axes)

        # Only show y-axis labels on leftmost column
        if col > 0:
            for ax in axes:
                ax.set_ylabel("")

    fig.suptitle(f"{controller_label} — trajectory tracking",
                 fontsize=13, y=1.01)

    tracking_path = f"{save_prefix}_tracking.png"
    plt.savefig(tracking_path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved → {tracking_path}")

    # Summary bar chart
    fig_sum = plot_summary(all_results, controller_label, color)
    summary_path = f"{save_prefix}_summary.png"
    fig_sum.savefig(summary_path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved → {summary_path}")