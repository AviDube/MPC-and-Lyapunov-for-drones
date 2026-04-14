"""
wind_robustness_demo.py
───────────────────────
Interactive MPC hover-robustness demo with live wind perturbations.

Layout
------
  Left  : MuJoCo passive viewer (3-D scene with professional lighting,
          ground-plane, and a live wind-vector arrow)
  Right : DearPyGui floating control panel

Controls
--------
  Wind X / Y / Z sliders  → model.opt.wind  (m/s, ±15)
  Zero Wind toggle         → snap all sliders to 0
  Reset Simulation button  → restart from origin

Requirements
------------
  pip install mujoco dearpygui scipy numpy

Run
---
  python wind_robustness_demo.py
  (place basic_quadrotor.xml + cf2.stl in the same directory, or set XML_PATH)
"""

import threading
import time

import numpy as np
import mujoco
import mujoco.viewer
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation
from mpc_hybrid import HybridMPC, get_state, wrench_to_rotors, USE_NN

# ═══════════════════════════════════════════════════════════════════════════════
# Paths & constants
# ═══════════════════════════════════════════════════════════════════════════════
XML_PATH = "../basic_quadrotor_presentation.xml"
DT_CTRL  = 0.02
nx, nu   = 12, 4

# Hover target
X_REF = np.zeros(nx)
X_REF[0] = 0.0
X_REF[1] = 0.0
X_REF[2] = 1.2   # 1.2 m altitude

# Wind-arrow geometry
ARROW_SHAFT_RADIUS = 0.012
MIN_ARROW_LEN      = 0.05   # always draw at least this long

# ═══════════════════════════════════════════════════════════════════════════════
# Load MuJoCo model
# ═══════════════════════════════════════════════════════════════════════════════
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)

# Inject fluid/aerodynamic properties so wind physically tilts the drone
mj_model.opt.density    = 1.225    # air density  [kg/m³]
mj_model.opt.viscosity  = 1.8e-5   # dynamic viscosity  [Pa·s]
mj_model.opt.wind[:]    = [0.0, 0.0, 0.0]

MASS   = float(mj_model.body_mass[1])
GRAV   = float(abs(mj_model.opt.gravity[2]))
INERTIA = tuple(mj_model.body_inertia[1])
HOVER  = MASS * GRAV / 4.0

# ═══════════════════════════════════════════════════════════════════════════════
# State helpers  (same convention as mpc_hybrid.py)
# ═══════════════════════════════════════════════════════════════════════════════
def quat_to_euler(q):
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_euler("xyz")

def get_state(d):
    return np.concatenate([
        d.qpos[:3],
        quat_to_euler(d.qpos[3:7]),
        d.qvel[:3],
        d.qvel[3:6],
    ])

def wrench_to_rotors(u):
    T, tx, ty, tz = u
    l = 0.028; k = 0.02513
    return np.array([
        T / 4 - tx / (4*l) - ty / (4*l) - tz / (4*k),
        T / 4 - tx / (4*l) + ty / (4*l) + tz / (4*k),
        T / 4 + tx / (4*l) + ty / (4*l) - tz / (4*k),
        T / 4 + tx / (4*l) - ty / (4*l) + tz / (4*k),
    ])

# ═══════════════════════════════════════════════════════════════════════════════
# Shared simulation state (thread-safe via simple flag + numpy arrays)
# ═══════════════════════════════════════════════════════════════════════════════
class SimState:
    def __init__(self):
        self.lock         = threading.Lock()
        self.wind         = np.zeros(3)       # requested wind
        self.reset_flag   = False
        self.running      = True              # set False to quit both threads
        self.drone_pos    = np.zeros(3)
        self.drone_err    = 0.0
        self.sim_time     = 0.0

sim_state = SimState()

# ═══════════════════════════════════════════════════════════════════════════════
# Wind-arrow visualisation helper
# ═══════════════════════════════════════════════════════════════════════════════
def _draw_wind_arrow(viewer, wind: np.ndarray, drone_pos: np.ndarray):
    """
    Draws a single, proper arrow in the MuJoCo scene-geometry overlay.
    Origin is slightly above the drone so it never clips into the body mesh.
    """
    mag = float(np.linalg.norm(wind))
    arrow_len = max(mag * 0.18, MIN_ARROW_LEN)  # scale with speed

    viewer.user_scn.ngeom = 0   # clear previous overlay geoms

    if mag < 1e-3:
        return   # no wind → nothing to draw

    direction = wind / mag          # unit vector

    origin = drone_pos + np.array([0.0, 0.0, 0.12])  # just above drone CoM
    
    # Midpoint of the arrow (MuJoCo primitives are defined by their center)
    arrow_mid = origin + direction * (arrow_len / 2.0)

    # ── build rotation matrix: local Z → direction ──────────────────────────
    z = direction
    # choose an arbitrary up that is not parallel to z
    up = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x  = np.cross(up, z); x /= np.linalg.norm(x)
    y  = np.cross(z, x)
    R  = np.column_stack([x, y, z])   # 3×3

    # Convert to 9-element row-major flat for mujoco.mjtMat
    mat = R.flatten()

    # ── draw built-in arrow ─────────────────────────────────────────────────
    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,  # Use the built-in arrow shape
        np.array([ARROW_SHAFT_RADIUS, ARROW_SHAFT_RADIUS, arrow_len / 2.0]),
        arrow_mid,
        mat,
        np.array([1.0, 0.55, 0.0, 0.90], dtype=np.float32),  # orange
    )
    viewer.user_scn.ngeom += 1

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation thread
# ═══════════════════════════════════════════════════════════════════════════════
def simulation_thread():
    global _integral

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[2] = 0.1   # start just off the ground
    mujoco.mj_forward(mj_model, mj_data)
    mpc = HybridMPC()

    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:

        # ── scene / camera tweaks for a professional look ──────────────────
        viewer.cam.distance  = 3.0
        viewer.cam.elevation = -25
        viewer.cam.azimuth   = 45
        viewer.cam.lookat[:] = [0.0, 0.0, 0.6]

        t_sim    = 0.0
        last_gui = time.time()

        while viewer.is_running():
            with sim_state.lock:
                if not sim_state.running:
                    break

                # Handle reset
                if sim_state.reset_flag:
                    mujoco.mj_resetData(mj_model, mj_data)
                    mj_data.qpos[2] = 0.1
                    mujoco.mj_forward(mj_model, mj_data)
                    t_sim              = 0.0
                    sim_state.reset_flag = False

                # Apply wind
                mj_model.opt.wind[:] = sim_state.wind

                current_wind = sim_state.wind.copy()

            # ── MPC / controller ──────────────────────────────────────────
            x = get_state(mj_data)
            u = mpc.solve(x, X_REF)
            u_rot=np.clip(wrench_to_rotors(u),0.005,0.25)
            mj_data.ctrl[:]=u_rot

            # ── Physics steps ─────────────────────────────────────────────
            n_steps = max(1, int(DT_CTRL / mj_model.opt.timestep))
            for _ in range(n_steps):
                mujoco.mj_step(mj_model, mj_data)

            t_sim += DT_CTRL

            # ── Update shared telemetry for GUI ───────────────────────────
            drone_pos = mj_data.qpos[:3].copy()
            pos_err   = float(np.linalg.norm(drone_pos - X_REF[:3]))
            with sim_state.lock:
                sim_state.drone_pos = drone_pos
                sim_state.drone_err = pos_err
                sim_state.sim_time  = t_sim

            # ── Wind arrow ────────────────────────────────────────────────
            _draw_wind_arrow(viewer, current_wind, drone_pos)

            viewer.sync()

    with sim_state.lock:
        sim_state.running = False


# ═══════════════════════════════════════════════════════════════════════════════
# GUI thread (DearPyGui)
# ═══════════════════════════════════════════════════════════════════════════════
_zero_wind_active = [False]

def gui_thread():
    dpg.create_context()

    # ── Theme ────────────────────────────────────────────────────────────────
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,      (22,  27,  34,  240))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,       (13,  110, 253, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  (13,  110, 253, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (40,  48,  60,  255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (55,  66,  82,  255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,     (13,  110, 253, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,(0,  80,  200, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button,         (40,  48,  60,  255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (13,  110, 253, 180))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (13,  110, 253, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text,           (220, 228, 240, 255))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark,      (13,  110, 253, 255))
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  10)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,    6)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,     6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,      8, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,     8, 5)

    # ── Red button theme ─────────────────────────────────────────────────────
    with dpg.theme() as red_btn_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (180,  30,  30, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (220,  50,  50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (150,  20,  20, 255))

    # ── Accent separator theme ───────────────────────────────────────────────
    with dpg.theme() as sep_theme:
        with dpg.theme_component(dpg.mvSeparator):
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (13, 110, 253, 180))

    dpg.create_viewport(
        title="Wind Robustness Demo — Control Panel",
        width=370, height=580,
        x_pos=20, y_pos=80,
        resizable=False,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    def on_wind_change(sender, value, user_data):
        if _zero_wind_active[0]:
            return
        axis = user_data  # 0, 1, or 2
        with sim_state.lock:
            sim_state.wind[axis] = value

    def on_zero_wind(sender, value):
        _zero_wind_active[0] = value
        if value:
            with sim_state.lock:
                sim_state.wind[:] = 0.0
            dpg.set_value("wind_x", 0.0)
            dpg.set_value("wind_y", 0.0)
            dpg.set_value("wind_z", 0.0)

    def on_reset():
        with sim_state.lock:
            sim_state.reset_flag = True

    # ── Main window ──────────────────────────────────────────────────────────
    with dpg.window(
        label="🚁  MPC Wind Robustness Demo",
        tag="main_win",
        width=354, height=560,
        no_resize=True, no_move=True,
        no_close=True, no_collapse=True,
    ):
        # Header
        dpg.add_text("Hover target: (0.0, 0.0, 1.2 m)", color=(160, 200, 255))
        dpg.add_separator(); dpg.add_spacer(height=4)

        # ── Telemetry ────────────────────────────────────────────────────────
        dpg.add_text("TELEMETRY", color=(100, 140, 200))
        dpg.bind_item_theme(dpg.last_item(), sep_theme)

        with dpg.group(horizontal=True):
            dpg.add_text("Sim time :", color=(160, 175, 195))
            dpg.add_text("0.00 s", tag="lbl_time", color=(220, 228, 240))

        with dpg.group(horizontal=True):
            dpg.add_text("Pos error:", color=(160, 175, 195))
            dpg.add_text("0.000 m", tag="lbl_err", color=(80, 220, 120))

        with dpg.group(horizontal=True):
            dpg.add_text("Drone pos:", color=(160, 175, 195))
            dpg.add_text("(0.00, 0.00, 0.00)", tag="lbl_pos", color=(220, 228, 240))

        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=4)

        # ── Wind control ─────────────────────────────────────────────────────
        dpg.add_text("WIND CONTROL  (m/s)", color=(100, 140, 200))

        dpg.add_spacer(height=4)
        dpg.add_text("Wind X  (East →)", color=(255, 180, 80))
        dpg.add_slider_float(
            tag="wind_x", min_value=-15.0, max_value=15.0,
            default_value=0.0, width=-1,
            callback=on_wind_change, user_data=0,
        )

        dpg.add_spacer(height=4)
        dpg.add_text("Wind Y  (North →)", color=(80, 200, 255))
        dpg.add_slider_float(
            tag="wind_y", min_value=-15.0, max_value=15.0,
            default_value=0.0, width=-1,
            callback=on_wind_change, user_data=1,
        )

        dpg.add_spacer(height=4)
        dpg.add_text("Wind Z  (Up →)", color=(120, 255, 160))
        dpg.add_slider_float(
            tag="wind_z", min_value=-15.0, max_value=15.0,
            default_value=0.0, width=-1,
            callback=on_wind_change, user_data=2,
        )

        dpg.add_spacer(height=6)
        with dpg.group(horizontal=True):
            dpg.add_text("Wind speed:", color=(160, 175, 195))
            dpg.add_text("0.0 m/s", tag="lbl_wind_mag", color=(255, 200, 80))

        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=6)

        # ── Actions ──────────────────────────────────────────────────────────
        dpg.add_text("ACTIONS", color=(100, 140, 200))
        dpg.add_spacer(height=6)

        dpg.add_checkbox(
            label=" Zero Wind  (freeze sliders at 0)",
            tag="chk_zero",
            callback=on_zero_wind,
        )

        dpg.add_spacer(height=8)
        btn_reset = dpg.add_button(
            label="↺  Reset Simulation",
            width=-1, height=42,
            callback=on_reset,
        )
        dpg.bind_item_theme(btn_reset, red_btn_theme)

        dpg.add_spacer(height=6)

        # Wind description legend
        dpg.add_separator()
        dpg.add_spacer(height=4)
        dpg.add_text(
            "Orange arrow in viewer = wind direction & magnitude",
            color=(140, 150, 165), wrap=340,
        )
        dpg.add_text(
            "Scales: 1 m/s ≈ 0.18 m arrow length",
            color=(120, 130, 148), wrap=340,
        )

    dpg.bind_theme(global_theme)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_win", True)

    # ── Render loop ──────────────────────────────────────────────────────────
    while dpg.is_dearpygui_running():
        # Check if sim exited
        with sim_state.lock:
            if not sim_state.running:
                break
            t      = sim_state.sim_time
            err    = sim_state.drone_err
            pos    = sim_state.drone_pos.copy()
            wind   = sim_state.wind.copy()

        # Update telemetry labels
        dpg.set_value("lbl_time", f"{t:.2f} s")

        err_col = (80, 220, 120) if err < 0.15 else \
                  (255, 200, 60) if err < 0.40 else (255, 80, 80)
        dpg.set_value("lbl_err", f"{err:.3f} m")
        dpg.configure_item("lbl_err", color=err_col)

        dpg.set_value("lbl_pos", f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        mag = float(np.linalg.norm(wind))
        dpg.set_value("lbl_wind_mag", f"{mag:.1f} m/s")

        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    with sim_state.lock:
        sim_state.running = False


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 58)
    print("  MPC Wind Robustness Demo")
    print("=" * 58)
    print(f"  Drone mass  : {MASS*1000:.1f} g")
    print(f"  Hover thrust: {HOVER*4:.4f} N  ({HOVER:.4f} N/rotor)")
    print(f"  Hover target: {X_REF[:3]}")
    print(f"  Air density : {mj_model.opt.density} kg/m³  (wind drag active)")
    print()
    print("  Launching MuJoCo viewer + DearPyGui control panel…")
    print("  Close either window to exit.")
    print("=" * 58)

    # GUI runs on main thread; simulation on background thread
    sim_thread = threading.Thread(target=simulation_thread, daemon=True)
    sim_thread.start()

    # Small delay so the viewer window can open first
    time.sleep(0.8)

    gui_thread()   # blocks until window is closed

    with sim_state.lock:
        sim_state.running = False

    sim_thread.join(timeout=3.0)
    print("Demo finished.")