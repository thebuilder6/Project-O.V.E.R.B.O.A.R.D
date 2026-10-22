"""Generate the vector figures used by docs/ControlsDoc.qmd.

Run from the repository root:

    python docs/figures/make_controlsdoc_figures.py

The document includes the SVGs as static images. Quarto is not asked to run
Python here: this environment has no Jupyter kernel, so an executable chunk
would fail the render.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "images" / "controlsdoc"
INK = "#1e293b"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#16a34a"
AMBER = "#d97706"


def save(fig, name):
    path = OUT / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    assert path.stat().st_size > 1000, path
    print(path)


def fig_kinematics_polytope():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.set_aspect("equal")
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title(r"(a) Chassis Frame $\mathcal{F}_B$ & Turning Geometry",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    x0, y0, theta = 0.5, 0.4, np.radians(35)
    B, L = 0.28, 0.36
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    chassis = np.array([[-L / 2, -B / 2], [L / 2, -B / 2],
                        [L / 2, B / 2], [-L / 2, B / 2],
                        [-L / 2, -B / 2]]).T
    chassis_rot = R @ chassis + np.array([[x0], [y0]])
    ax1.plot(chassis_rot[0], chassis_rot[1], color=INK, lw=2, label="Chassis body")
    ax1.fill(chassis_rot[0], chassis_rot[1], color="#cbd5e1", alpha=0.4)

    wheel_len, wheel_w = 0.12, 0.04
    for sign, colour, name in [(-1, RED, r"$v_l$"), (1, GREEN, r"$v_r$")]:
        box = np.array([[-wheel_len / 2, sign * B / 2 - wheel_w / 2],
                        [wheel_len / 2, sign * B / 2 - wheel_w / 2],
                        [wheel_len / 2, sign * B / 2 + wheel_w / 2],
                        [-wheel_len / 2, sign * B / 2 + wheel_w / 2],
                        [-wheel_len / 2, sign * B / 2 - wheel_w / 2]]).T
        box_rot = R @ box + np.array([[x0], [y0]])
        ax1.fill(box_rot[0], box_rot[1], color="#0f172a", zorder=4)
        vec_len = 0.25 if sign == 1 else 0.15
        start = R @ np.array([0, sign * B / 2]) + np.array([x0, y0])
        direction = R @ np.array([vec_len, 0])
        ax1.annotate("", xy=start + direction, xytext=start,
                     arrowprops=dict(arrowstyle="->", color=colour, lw=2.2))
        ax1.text(start[0] + direction[0] + 0.02, start[1] + direction[1], name,
                 color=colour, fontweight="bold", fontsize=10)

    ax1.annotate("", xy=np.array([x0, y0]) + R @ np.array([0.22, 0]),
                 xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax1.text(x0 + 0.22 * np.cos(theta), y0 + 0.22 * np.sin(theta) + 0.03,
             r"$X_B$", color=BLUE, fontweight="bold")
    ax1.plot(x0, y0, "ko", markersize=5)
    ax1.text(x0 - 0.08, y0 - 0.08, r"$(x, y, \theta)$", fontsize=9)

    # vr > vl, so the ICC sits on the left of the body x-axis.
    icc_dist = 0.14 * (0.15 + 0.25) / (0.25 - 0.15)
    icc = np.array([x0, y0]) + R @ np.array([0, icc_dist])
    ax1.plot(icc[0], icc[1], "r*", markersize=10, label="ICC")
    ax1.plot([x0, icc[0]], [y0, icc[1]], "r:", lw=1.5)
    ax1.text(icc[0] + 0.03, icc[1] - 0.02, "ICC", color=RED,
             fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.5)

    v_max = 1.2
    omega_max = 2 * v_max / B
    ax2.set_xlim(-omega_max * 1.15, omega_max * 1.15)
    ax2.set_ylim(-v_max * 1.2, v_max * 1.2)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title(r"(b) Actuator Diamond in $(v, \omega)$ Space",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Yaw rate $\omega$ (rad/s)")
    ax2.set_ylabel(r"Linear speed $v$ (m/s)")

    omega_poly = np.array([0, omega_max, 0, -omega_max, 0])
    v_poly = np.array([v_max, 0, -v_max, 0, v_max])
    ax2.plot(omega_poly, v_poly, color="#0284c7", lw=2.5,
             label="Actuator diamond boundary")
    ax2.fill(omega_poly, v_poly, color="#38bdf8", alpha=0.25)
    ax2.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax2.axvline(0, color="black", lw=0.8, alpha=0.6)

    edge_angle = np.degrees(np.arctan2(-v_max, omega_max))
    ax2.text(omega_max * 0.34, 0.88, r"$v_r = +v_{\max}$",
             rotation=edge_angle, color="#0369a1", fontweight="bold",
             fontsize=9, ha="center")
    ax2.text(-omega_max * 0.34, 0.88, r"$v_l = +v_{\max}$",
             rotation=-edge_angle, color="#0369a1", fontweight="bold",
             fontsize=9, ha="center")
    ax2.plot(0, v_max, "ro", markersize=6)
    ax2.text(0.25, v_max + 0.04, r"$v_{\max}$ (pure translation)",
             fontsize=9, color="#b91c1c", ha="left")
    ax2.plot(omega_max, 0, "go", markersize=6)
    ax2.text(omega_max * 0.55, -0.22,
             r"$\omega_{\max} = 2v_{\max}/B$", fontsize=9, color="#15803d",
             ha="center")
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    save(fig, "kinematics_polytope")


def fig_late_apex():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    u = np.linspace(0, 1, 300)
    u_peak, a = 0.68, 2.0
    b = a * (1 - u_peak) / u_peak
    norm = (u_peak ** a) * ((1 - u_peak) ** b)
    w_late = np.where((u > 0) & (u < 1),
                      (u ** a) * ((1 - u) ** b) / norm, 0.0)
    w_symm = 4.0 * u * (1 - u)

    ax1.plot(u, w_late, color=AMBER, lw=2.5,
             label=r"Late-apex bump ($u_{\mathrm{peak}}=0.68$)")
    ax1.plot(u, w_symm, "k--", lw=1.5, alpha=0.6,
             label="Symmetric parabolic ($u=0.50$)")
    ax1.axvline(u_peak, color=AMBER, linestyle=":", lw=1.5)
    ax1.axvline(0.50, color="black", linestyle=":", lw=1.2, alpha=0.6)
    ax1.text(u_peak + 0.02, 0.86, r"$u_{\mathrm{peak}} = 0.68$",
             color=AMBER, fontweight="bold", fontsize=9)
    ax1.text(0.32, 0.86, "$u = 0.50$", color="#475569", fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.08)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title(r"(a) Asymmetric Weighting $w_{\mathrm{apex}}(u)$",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Normalized segment coordinate $u$")
    ax1.set_ylabel("Normalized lateral bump $w(u)$")
    ax1.legend(loc="lower center", fontsize=9)

    w0, w1, w2 = (np.array([0.0, 0.0]), np.array([0.8, 0.0]),
                  np.array([0.8, 0.8]))
    ax2.plot([w0[0], w1[0], w2[0]], [w0[1], w1[1], w2[1]], "k--",
             lw=1.5, alpha=0.5, label="Direct waypoint chords")

    amp = 0.08
    s1 = np.array([w0 + t * (w1 - w0) for t in u]) \
        - amp * w_late[:, None] * np.array([0.0, 1.0])
    s2 = np.array([w1 + t * (w2 - w1) for t in u]) \
        - amp * w_late[:, None] * np.array([-1.0, 0.0])
    path = np.vstack([s1, s2])
    ax2.plot(path[:, 0], path[:, 1], color=BLUE, lw=2.5,
             label="Late-apex analytical seed")
    ax2.plot([w0[0], w1[0], w2[0]], [w0[1], w1[1], w2[1]], "ro",
             markersize=6, zorder=5)
    ax2.text(w0[0], w0[1] - 0.06, "$W_0$", fontweight="bold", ha="center")
    ax2.text(w1[0] + 0.04, w1[1] - 0.05, "$W_1$ (corner)",
             fontweight="bold")
    ax2.text(w2[0] + 0.04, w2[1], "$W_2$", fontweight="bold")

    idx = int(u_peak * len(u))
    ax2.plot(s1[idx, 0], s1[idx, 1], "m*", markersize=10, zorder=6,
             label="Late apex point")
    ax2.annotate("Deep trail-braking\nentry", xy=(0.35, -0.04),
                 xytext=(0.12, 0.16),
                 arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5),
                 fontsize=8)
    ax2.annotate("Early straight exit", xy=(s2[135, 0], s2[135, 1]),
                 xytext=(0.42, 0.55),
                 arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5),
                 fontsize=8)

    ax2.set_aspect("equal")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title(r"(b) Corner Transition ($A_{\mathrm{apex}} = 80$ mm)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    save(fig, "late_apex")


def fig_velocity_ceiling_energy():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    kappa = np.linspace(0.01, 12, 400)
    v_wheel_max, B, h_cg, mu, g = 1.05, 0.28, 0.05, 0.6, 9.81
    v_frontier = v_wheel_max / (1.0 + 0.5 * B * kappa)
    v_tip = np.sqrt((g * B) / (2.0 * h_cg * kappa))
    v_trac = np.sqrt(mu * g / kappa)
    v_ceil = np.minimum(v_wheel_max,
                        np.minimum(v_frontier, np.minimum(v_tip, v_trac)))

    ax1.plot(kappa, np.full_like(kappa, v_wheel_max), "k--", lw=1.2,
             label=r"Actuator top speed $v_{\mathrm{wheel,max}}$")
    ax1.plot(kappa, v_frontier, color="#0284c7", lw=1.8,
             label=r"Actuator diamond $v_{\mathrm{frontier}}$")
    ax1.plot(kappa, v_tip, color="#ea580c", lw=1.8, linestyle="-.",
             label=r"Wheel-lift ceiling $v_{\mathrm{tip}}$")
    ax1.plot(kappa, v_trac, color=GREEN, lw=1.8, linestyle=":",
             label=r"Traction ceiling $v_{\mathrm{traction}}$")
    ax1.plot(kappa, v_ceil, color="#b91c1c", lw=3.0, alpha=0.9,
             label=r"Composite ceiling $v_{\mathrm{ceiling}}(\kappa)$")
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0.1, 100)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.set_title(r"(a) Velocity ceilings vs. curvature $\kappa$",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Path curvature $|\kappa|$ (rad/m)")
    ax1.set_ylabel("Velocity ceiling (m/s, log scale)")
    ax1.legend(loc="upper right", fontsize=8.5)

    s = np.linspace(0, 2.5, 250)
    k_prof = (0.5 + 4.5 * np.exp(-((s - 0.7) / 0.15) ** 2)
              + 7.0 * np.exp(-((s - 1.8) / 0.2) ** 2))
    v_prof = np.minimum(v_wheel_max, v_wheel_max / (1.0 + 0.5 * B * k_prof))
    E_ceil = v_prof ** 2
    E_ceil[0] = 0.0
    E_ceil[-1] = 0.0

    a_max = 1.8
    ds = s[1] - s[0]
    dE = 2.0 * a_max * ds

    E_fwd = np.copy(E_ceil)
    for k in range(len(s) - 1):
        E_fwd[k + 1] = min(E_fwd[k + 1], E_fwd[k] + dE)

    E_bwd = np.copy(E_fwd)
    for k in range(len(s) - 2, -1, -1):
        E_bwd[k] = min(E_bwd[k], E_bwd[k + 1] + dE)

    ax2.plot(s, np.sqrt(E_ceil), "k:", lw=1.5,
             label=r"Corner ceiling $v_{\mathrm{ceiling}}$")
    ax2.plot(s, np.sqrt(E_fwd), color="#0284c7", linestyle="--", lw=1.8,
             label="Forward acceleration pass")
    ax2.plot(s, np.sqrt(E_bwd), color=RED, lw=2.5,
             label="Backward braking pass (seed)")
    ax2.fill_between(s, 0, np.sqrt(E_bwd), color="#fca5a5", alpha=0.3)

    ax2.plot(0, 0, "go", markersize=6)
    ax2.plot(s[-1], 0, "ro", markersize=6)
    ax2.text(0.05, 0.1, "Start ($v=0$)", fontsize=8.5, color="#15803d",
             fontweight="bold")
    ax2.text(s[-1] - 0.45, 0.1, "Stop ($v=0$)", fontsize=8.5,
             color="#b91c1c", fontweight="bold")
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title(r"(b) Dual-pass energy integration ($E = v^2$)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Arc distance $s$ (m)")
    ax2.set_ylabel("Velocity $v$ (m/s)")
    ax2.legend(loc="lower center", fontsize=8.5)

    fig.tight_layout()
    save(fig, "velocity_ceiling_energy")


def fig_hessian_epigraph():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    dF = np.linspace(-5e-6, 5e-6, 400)
    eps = 1e-12
    d2F = eps / ((dF ** 2 + eps) ** 1.5)
    ax1.semilogy(dF * 1e6, d2F, color=RED, lw=2.2,
                 label=r"$\partial^2/\partial F^2\,\sqrt{F^2+10^{-12}}$")
    ax1.axvline(0, color="black", linestyle=":", lw=1.2, alpha=0.7)
    ax1.annotate(r"Singular spike: $\sim 10^6$", xy=(0, np.max(d2F)),
                 xytext=(-4.4, 2e4),
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.5),
                 fontsize=9, fontweight="bold")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(1e0, 2e6)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.set_title(r"(a) Hessian singularity at $a_l = a_r$",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Smoothing argument $F$ ($\mu$m)")
    ax1.set_ylabel(r"Second derivative (log scale)")
    ax1.legend(loc="lower center", fontsize=9)

    f_lin = np.linspace(-10, 10, 100)
    f_ang = np.linspace(-10, 10, 100)
    FL, FA = np.meshgrid(f_lin, f_ang)
    fb_level = np.maximum(np.abs(FL - FA), np.abs(FL + FA))
    cs = ax2.contour(FL, FA, fb_level, levels=[2, 4, 6, 8, 10],
                     cmap="Blues_r")
    ax2.clabel(cs, inline=True, fontsize=8, fmt="f_b=%.0f N")

    region = np.array([[-4, -4], [0, -8], [4, -4], [8, 0],
                       [4, 4], [0, 8], [-4, 4], [-8, 0], [-4, -4]])
    ax2.plot(region[:, 0], region[:, 1], color="#0284c7", lw=2,
             label=r"Feasible slice ($f_{b,k} \leq 8$ N)")
    ax2.fill(region[:, 0], region[:, 1], color="#38bdf8", alpha=0.25)
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title(r"(b) Linear epigraph bounding polyhedron",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Longitudinal component $f_{\mathrm{lin}}$ (N)")
    ax2.set_ylabel(r"Yaw-moment component $f_{\mathrm{ang}}$ (N)")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    save(fig, "hessian_epigraph")


def fig_discretization_mechanisms():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.8))

    t_arc = np.linspace(-0.6, 0.6, 101)
    R_curve = 0.8
    x_arc = R_curve * np.sin(t_arc)
    y_arc = R_curve * np.cos(t_arc) - R_curve * np.cos(0.6)
    ax1.plot(x_arc, y_arc, color=BLUE, lw=2.5, label="Circular arc")
    ax1.plot([x_arc[0], x_arc[-1]], [y_arc[0], y_arc[-1]], "k--", lw=2,
             label="Linear chord")
    ax1.plot([0, 0], [0, y_arc[50]], "r-", lw=2.2)
    ax1.plot(0, y_arc[50], "ro", markersize=5)
    ax1.text(0.04, y_arc[50] / 2,
             r"$e_{\mathrm{sagitta}} = \frac{1}{8}v\omega\Delta t^2$",
             color=RED, fontweight="bold", fontsize=9)
    ax1.set_title(r"(a) Chord vs. arc sagitta", fontsize=10.5,
                  fontweight="bold")
    ax1.axis("equal")
    ax1.set_ylim(-0.05, 0.25)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower center", fontsize=8)

    t_step = np.linspace(0, 1, 100)
    theta_linear = 0.5 * t_step
    theta_parab = 0.5 * t_step + 0.5 * 0.8 * (t_step ** 2)
    ax2.plot(t_step, theta_parab, color="#0891b2", lw=2.5,
             label=r"True angle $\theta(t)$")
    ax2.plot(t_step, theta_linear, "k--", lw=2, label="Linear approximation")
    ax2.fill_between(t_step, theta_linear, theta_parab,
                     color="#06b6d4", alpha=0.25)
    ax2.text(0.35, 0.45,
             r"$e_{\mathrm{parab}} = \frac{1}{8}v\Delta\omega\Delta t^2$",
             color="#0e7490", fontweight="bold", fontsize=9)
    ax2.set_title(r"(b) Parabolic heading mismatch", fontsize=10.5,
                  fontweight="bold")
    ax2.set_xlabel("Normalized interval time $s$")
    ax2.set_ylabel(r"Heading $\theta$ (rad)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)

    t_hold = np.array([0, 0.4, 0.4, 0.8, 0.8, 1.0])
    v_profile = np.array([0.4, 0.4, 0.7, 0.7, 0.95, 0.95])
    t_ideal = np.linspace(0, 1.0, 100)
    v_ideal = 0.4 + 0.55 * t_ideal
    ax3.step(t_hold, v_profile, where="post", color=AMBER, lw=2.2,
             label="20 ms ZOH output")
    ax3.plot(t_ideal, v_ideal, "k--", lw=1.8, label="Optimal velocity")
    ax3.fill_between(t_hold, v_profile, 0.4 + 0.55 * t_hold,
                     color="#fde68a", alpha=0.5)
    ax3.text(0.12, 0.75,
             r"$e_{\mathrm{hold}} = \frac{1}{2}\Delta v\Delta t_{\mathrm{ctrl}}$",
             color="#b45309", fontweight="bold", fontsize=9)
    ax3.set_title(r"(c) 20 ms sample-and-hold lag", fontsize=10.5,
                  fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Velocity $v$ (m/s)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    save(fig, "discretization_mechanisms")


def fig_reachability_tube():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    t_sim = np.linspace(0, 3.5, 300)
    x_ref = 0.8 * np.sin(t_sim * 1.2)
    y_ref = t_sim * 0.5
    tube_rad = 0.006 + 0.003 * np.sin(t_sim * 2.0) ** 2 + 0.0015 * t_sim
    track_err = 0.002 + 0.002 * np.sin(t_sim * 1.8) ** 2
    assert tube_rad.max() < 0.020
    assert track_err.max() < 0.005

    ax1.plot(x_ref, y_ref, color=BLUE, lw=2.2,
             label=r"Nominal plan $\mathbf{p}_{\mathrm{ref}}$")
    ax1.plot(x_ref + track_err, y_ref, color="#0891b2", lw=1.5,
             linestyle="--", label=r"Tracking centroid $\mathbf{p}_{\mathrm{mid}}$")
    ax1.fill_betweenx(y_ref, x_ref - tube_rad, x_ref + tube_rad,
                      color="#34d399", alpha=0.35,
                      label=r"Reachability tube $[\mathbf{x}_k]$")
    ax1.plot(x_ref - tube_rad, y_ref, color="#059669", lw=1.0,
             linestyle=":")
    ax1.plot(x_ref + tube_rad, y_ref, color="#059669", lw=1.0,
             linestyle=":")
    ax1.annotate("Tracking error ≤ 5 mm",
                 xy=(x_ref[120] + track_err[120], y_ref[120]),
                 xytext=(x_ref[120] + 0.08, y_ref[120] - 0.2),
                 arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.5),
                 fontsize=8.5, fontweight="bold")
    ax1.set_xlim(-1.2, 1.2)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title(r"(a) Reachability tube ($r_{\mathrm{tube}} \leq 20$ mm)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.legend(loc="upper left", fontsize=8.5)

    t_steps = np.linspace(0, 3.5, 60)
    grad_norm = (0.2 + 8.5 * np.exp(-((t_steps - 1.4) / 0.25) ** 2)
                 + 12.0 * np.exp(-((t_steps - 2.8) / 0.3) ** 2))
    weights = np.clip(1.0 + 4.0 * grad_norm / np.max(grad_norm), 1.0, 200.0)
    ax2.plot(t_steps, grad_norm, color=RED, lw=2,
             label=r"Adjoint gradient $\|\mathbf{g}_k\|_2$")
    ax2.step(t_steps, weights, where="mid", color=AMBER, lw=1.8,
             linestyle="--", label=r"Collocation weight $W_k$")
    ax2.set_xlim(0, 3.5)
    ax2.set_ylim(0, 15)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title(r"(b) Adjoint sensitivities and weights",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Trajectory time $t$ (s)")
    ax2.set_ylabel(r"Gradient magnitude / weight $W_k$")
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    save(fig, "reachability_tube")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_kinematics_polytope()
    fig_late_apex()
    fig_velocity_ceiling_energy()
    fig_hessian_epigraph()
    fig_discretization_mechanisms()
    fig_reachability_tube()
