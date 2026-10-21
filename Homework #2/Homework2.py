"""MAE-263F Homework 2 simulator.

Run `python Homework2.py` for the question results and plots. The functions
`simulate_falling_beam` and `simulate_loaded_beam` are also importable for
custom parameter studies.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from gradEb import gradEb
from hessEb import hessEb
from gradEs import gradEs
from hessEs import hessEs


def elastic_forces_and_jacobian(q: np.ndarray, EI: float, EA: float,
                                deltaL: float) -> tuple[np.ndarray, np.ndarray]:
    """Assemble elastic force (-energy gradient) and its Jacobian."""
    nv = q.size // 2
    F = np.zeros_like(q)
    J = np.zeros((q.size, q.size))

    # N-2 bending springs, one centered at each interior node.
    for k in range(1, nv - 1):
        idx = np.arange(2 * k - 2, 2 * k + 4)
        F[idx] -= gradEb(*q[idx], 0.0, deltaL, EI)
        J[np.ix_(idx, idx)] -= hessEb(*q[idx], 0.0, deltaL, EI)

    # N-1 stretching springs, one on each edge.
    for k in range(nv - 1):
        idx = np.arange(2 * k, 2 * k + 4)
        F[idx] -= gradEs(*q[idx], deltaL, EA)
        J[np.ix_(idx, idx)] -= hessEs(*q[idx], deltaL, EA)

    return F, J


def _newton_step(q_guess: np.ndarray, q_old: np.ndarray, u_old: np.ndarray,
                 dt: float, mass_dof: np.ndarray, EI: float, EA: float,
                 deltaL: float, external_force: np.ndarray,
                 damping_diag: np.ndarray | None = None,
                 fixed_index: np.ndarray | None = None,
                 tolerance: float = 1e-8,
                 maximum_iter: int = 100) -> tuple[np.ndarray, int, float]:
    """Implicit Euler step using Newton's method, optionally with constraints."""
    q_new = q_guess.copy()
    ndof = q_new.size
    if fixed_index is None:
        fixed_index = np.array([], dtype=int)
    free_index = np.setdiff1d(np.arange(ndof), fixed_index)
    inv_dt2 = 1.0 / (dt * dt)
    Mdiag = mass_dof * inv_dt2
    if damping_diag is None:
        damping_diag = np.zeros(ndof)

    for iteration in range(1, maximum_iter + 1):
        Felastic, Jelastic = elastic_forces_and_jacobian(q_new, EI, EA, deltaL)
        Fi = mass_dof * ((q_new - q_old) * inv_dt2 - u_old / dt)
        # Viscous force is -C * (q_new-q_old)/dt. The residual subtracts
        # physical forces, giving a +C/dt contribution to the Jacobian.
        Fvisc = -damping_diag * (q_new - q_old) / dt
        residual = Fi - Felastic - Fvisc - external_force
        J = np.diag(Mdiag + damping_diag / dt) - Jelastic
        residual_free = residual[free_index]
        error = float(np.linalg.norm(residual_free))
        if error <= tolerance:
            return q_new, iteration, error
        J_free = J[np.ix_(free_index, free_index)]
        try:
            correction = np.linalg.solve(J_free, residual_free)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Newton Jacobian is singular.") from exc
        q_new[free_index] -= correction

    raise RuntimeError(
        f"Newton did not converge in {maximum_iter} iterations; residual={error:.4g}."
    )


def simulate_falling_beam(nv: int = 21, dt: float = 0.01,
                          totalTime: float = 50.0,
                          plotShapes: bool = False, visc: float = 1000.0,
                          R_all: float | None = None) -> dict[str, np.ndarray | float]:
    """Simulate the elastic beam falling through viscous fluid (Part 1)."""
    if nv < 3 or nv % 2 == 0:
        raise ValueError("nv must be an odd integer >= 3.")
    if dt <= 0 or totalTime < 0 or visc < 0:
        raise ValueError("dt must be positive; totalTime and visc must be nonnegative.")

    L = 0.10
    deltaL = L / (nv - 1)
    mid_node = (nv - 1) // 2
    radii = np.full(nv, deltaL / 10.0)
    radii[mid_node] = 0.025
    if R_all is not None:
        radii[:] = float(R_all)

    rho_metal, rho_fluid = 7000.0, 1000.0
    rod_radius = 1e-3
    E = 1e9
    EI = E * np.pi * rod_radius**4 / 4.0
    EA = E * np.pi * rod_radius**2

    mass_nodes = (4.0 / 3.0) * np.pi * radii**3 * rho_metal
    mass_dof = np.repeat(mass_nodes, 2)
    W = np.zeros(2 * nv)
    W[1::2] = -(4.0 / 3.0) * np.pi * radii**3 * (rho_metal - rho_fluid) * 9.81
    external_force = W
    Cdiag = np.repeat(6.0 * np.pi * visc * radii, 2)

    q_old = np.zeros(2 * nv)
    q_old[0::2] = np.arange(nv) * deltaL
    u_old = np.zeros(2 * nv)
    nsteps = round(totalTime / dt)
    times = np.arange(nsteps + 1) * dt
    y_mid = np.zeros(nsteps + 1)
    v_mid = np.zeros(nsteps + 1)
    tol = EI / (L * L) * 1e-6

    for step in range(1, nsteps + 1):
        q_new, _, _ = _newton_step(
            q_old, q_old, u_old, dt, mass_dof, EI, EA, deltaL,
            external_force, damping_diag=Cdiag, tolerance=tol,
        )
        u_new = (q_new - q_old) / dt
        q_old = q_new
        u_old = u_new
        y_mid[step] = q_new[2 * mid_node + 1]
        v_mid[step] = u_new[2 * mid_node + 1]
        if plotShapes and step % 10 == 0:
            plt.figure(1)
            plt.clf()
            plt.plot(q_new[0::2], q_new[1::2], "o-")
            plt.title(f"Falling beam at t={times[step]:.2f} s")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.axis("equal")
            plt.pause(0.01)

    return {"t": times, "y_mid": y_mid, "v_mid": v_mid,
            "q_final": q_old, "u_final": u_old, "radii": radii}


def simulate_loaded_beam(P: float, N: int = 50, dt: float = 0.01,
                         totalTime: float = 1.0, plot: bool = False,
                         tolerance_factor: float = 1e-3) -> dict[str, np.ndarray | float | int]:
    """Simulate a simply supported tubular aluminum beam (Part 2)."""
    if N < 3 or P < 0 or dt <= 0 or totalTime < 0:
        raise ValueError("Require N >= 3, P >= 0, dt > 0, and totalTime >= 0.")

    L = 1.0
    outer_radius, inner_radius = 0.013, 0.011
    E, rho = 70e9, 2700.0
    area = np.pi * (outer_radius**2 - inner_radius**2)
    EI = E * np.pi / 4.0 * (outer_radius**4 - inner_radius**4)
    EA = E * area
    deltaL = L / (N - 1)
    # Lumped mass per node; duplicated for x and y degrees of freedom.
    m = area * L * rho / (N - 1)
    mass_dof = np.full(2 * N, m)

    q_old = np.zeros(2 * N)
    q_old[0::2] = np.arange(N) * deltaL
    u_old = np.zeros(2 * N)

    # Apply the point load to the node closest to d=0.75 m.
    d = 0.75
    loaded_node = int(np.argmin(np.abs(q_old[0::2] - d)))
    external_force = np.zeros(2 * N)
    external_force[2 * loaded_node + 1] = -P

    fixed_index = np.array([0, 1, 2 * N - 1], dtype=int)  # x_1, y_1, y_N
    nsteps = round(totalTime / dt)
    times = np.arange(nsteps + 1) * dt
    ymax = np.zeros(nsteps + 1)
    x_at_ymax = np.full(nsteps + 1, q_old[0::2][0])
    tol = tolerance_factor * EI / (L * L)
    iteration_counts = np.zeros(nsteps, dtype=int)

    for step in range(1, nsteps + 1):
        q_new, iterations, _ = _newton_step(
            q_old, q_old, u_old, dt, mass_dof, EI, EA, deltaL,
            external_force, damping_diag=None, fixed_index=fixed_index,
            tolerance=tol, maximum_iter=100,
        )
        u_new = (q_new - q_old) / dt
        ys = q_new[1::2]
        imin = int(np.argmin(ys))
        ymax[step] = ys[imin]
        x_at_ymax[step] = q_new[2 * imin]
        iteration_counts[step - 1] = iterations
        q_old, u_old = q_new, u_new

        if plot and step % max(1, nsteps // 10) == 0:
            print(f"  t={times[step]:.2f} s, min(y)={ymax[step]:.6g} m")

    if plot:
        plt.figure()
        plt.plot(times, ymax)
        plt.xlabel("Time [s]")
        plt.ylabel("Minimum y-coordinate [m]")
        plt.title(f"Loaded beam: P={P:g} N")
        plt.tight_layout()
        plt.savefig(Path(f"loaded_beam_time_P_{int(P)}.png"), dpi=160)
        plt.close()
        plt.figure()
        plt.plot(q_old[0::2], q_old[1::2], "o-")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.title(f"Final beam shape: P={P:g} N")
        plt.tight_layout()
        plt.savefig(Path(f"loaded_beam_shape_P_{int(P)}.png"), dpi=160)
        plt.close()

    return {"t": times, "ymax": ymax, "x_at_ymax": x_at_ymax,
            "q_final": q_old, "u_final": u_old, "EI": EI, "EA": EA,
            "mass_per_node": m, "loaded_node_zero_based": loaded_node,
            "loaded_x": q_old[2 * loaded_node],
            "iterations": iteration_counts}


def buoyant_free_fall_velocity(t: float = 1.0) -> float:
    """Analytical Part 1 Q1 value (equal sphere radii and no drag)."""
    g, rho_metal, rho_fluid = 9.81, 7000.0, 1000.0
    return -g * (1.0 - rho_fluid / rho_metal) * t


def stokes_sphere_velocity(R: float = 0.025, visc: float = 1000.0) -> float:
    """Analytical terminal velocity of one sphere from Stokes' law."""
    return -(2.0 * R**2 * (7000.0 - 1000.0) * 9.81) / (9.0 * visc)


def terminal_beam_velocity(N: int, L: float = 0.1, Rmid: float = 0.025,
                           visc: float = 1000.0) -> float:
    """Analytical terminal velocity from force balance for unequal sphere radii."""
    small_R = L / (10.0 * (N - 1))
    sum_R = Rmid + (N - 1) * small_R
    sum_R3 = Rmid**3 + (N - 1) * small_R**3
    return -(2.0 * (7000.0 - 1000.0) * 9.81 / (9.0 * visc)) * sum_R3 / sum_R


def euler_bernoulli_deflection(P: float, d: float = 0.75, L: float = 1.0,
                               EI: float = 765.29) -> float:
    """Analytical maximum deflection magnitude from the homework's Eq. (6)."""
    c = min(d, L - d)
    return P * c * (L**2 - c**2)**1.5 / (9.0 * np.sqrt(3.0) * EI * L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1-full", action="store_true",
                        help="also run the slower Part 1 numerical checks")
    parser.add_argument("--no-plots", action="store_true",
                        help="do not save Part 2 diagnostic plots")
    parser.add_argument("--sweep", action="store_true",
                        help="run a denser Part 2 load sweep and save comparison plot")
    args = parser.parse_args()

    print("PART 1: analytical checks (negative velocity means downward)")
    v1 = buoyant_free_fall_velocity(1.0)
    v2 = stokes_sphere_velocity()
    v3 = terminal_beam_velocity(21)
    y4 = v3 * 50.0
    print(f"Q1: v_mid(1 s) = {v1:.6g} m/s -> choice (c), -8.4 m/s")
    print(f"Q2: v_mid(1 s) = {v2:.6g} m/s -> choice (b), -8.17e-3 m/s")
    print(f"Q3: terminal v, N=21 = {v3:.6g} m/s -> choice (d), -5.83e-3 m/s")
    print(f"Q4: approximate y_mid(50 s) = {y4:.6g} m -> choice (c), -0.29 m")
    vals = [(N, abs(terminal_beam_velocity(N))) for N in (3, 5, 11, 21)]
    print("Q5: |terminal velocity| by N: " + ", ".join(f"N={N}: {v:.6g}" for N, v in vals)
          + " m/s -> choice (a), decreases slightly")

    if args.part1_full:
        print("\nRunning Part 1 numerical checks; this can take several minutes.")
        one = simulate_falling_beam(nv=21, dt=0.01, totalTime=1.0,
                                    visc=0.0, R_all=0.025)
        two = simulate_falling_beam(nv=21, dt=0.01, totalTime=1.0,
                                    visc=1000.0, R_all=0.025)
        three = simulate_falling_beam(nv=21, dt=0.01, totalTime=50.0,
                                      visc=1000.0, R_all=None)
        print(f"  numerical Q1: {one['v_mid'][-1]:.6g} m/s")
        print(f"  numerical Q2: {two['v_mid'][-1]:.6g} m/s")
        print(f"  numerical Q3: {three['v_mid'][-1]:.6g} m/s")
        print(f"  numerical Q4: {three['y_mid'][-1]:.6g} m")
        for N in (3, 5, 11, 21):
            out = simulate_falling_beam(nv=N, dt=0.01, totalTime=50.0,
                                         visc=1000.0, R_all=None)
            print(f"  numerical Q5 N={N}: {out['v_mid'][-1]:.6g} m/s")

    print("\nPART 2: implicit-Euler simulations (N=50, dt=0.01 s, t_end=1 s)")
    outputs = {}
    for P in (2000.0, 20000.0):
        result = simulate_loaded_beam(P, plot=not args.no_plots)
        outputs[P] = result
        deflection = abs(float(result["ymax"][-1]))
        xloc = float(result["x_at_ymax"][-1])
        eb = euler_bernoulli_deflection(P, EI=float(result["EI"]))
        print(f"P={P:,.0f} N: |ymax|={deflection:.6g} m, x_at_ymax={xloc:.6g} m, "
              f"EB={eb:.6g} m, max Newton iterations={int(np.max(result['iterations']))}")
    q2000 = outputs[2000.0]
    print(f"Q6: P=2000 N deflection = {abs(float(q2000['ymax'][-1])):.6g} m")
    print(f"Q7: max-deflection location = {float(q2000['x_at_ymax'][-1]):.6g} m")
    print("Q8: choice (b) - implicit Euler adds numerical damping and settles quickly.")
    q20000 = outputs[20000.0]
    print(f"Q9: P=20000 N deflection = {abs(float(q20000['ymax'][-1])):.6g} m")

    candidates = [1000.0, 2000.0, 4000.0, 8000.0]
    errs = []
    print("\nQ10 relative difference e(P) = (EB - simulation) / EB:")
    for P in candidates:
        res = outputs.get(P)
        if res is None:
            res = simulate_loaded_beam(P, plot=False)
        ysim = abs(float(res["ymax"][-1]))
        yeb = euler_bernoulli_deflection(P, EI=float(res["EI"]))
        error = (yeb - ysim) / yeb
        errs.append((P, error))
        print(f"  P={P:,.0f} N: simulation={ysim:.6g} m, EB={yeb:.6g} m, e={error:.2%}")
    first = next((P for P, e in errs if e > 0.10), None)
    if first is not None:
        print(f"Q10: first candidate with e(P)>10% is {first:,.0f} N")
    else:
        print("Q10: none of the four candidates exceeded 10%; run --sweep to inspect more loads")
    print("Q11: choice (a) - Euler-Bernoulli theory assumes small slopes and undeformed geometry.")

    if args.sweep:
        loads = np.geomspace(20.0, 20000.0, 18)
        sim_values, eb_values = [], []
        for P in loads:
            result = simulate_loaded_beam(float(P), plot=False)
            sim_values.append(abs(float(result["ymax"][-1])))
            eb_values.append(euler_bernoulli_deflection(float(P), EI=float(result["EI"])))
        plt.figure()
        plt.loglog(loads, sim_values, "o-", label="Simulation")
        plt.loglog(loads, eb_values, "--", label="Euler-Bernoulli")
        plt.xlabel("Point load P [N]")
        plt.ylabel("Maximum deflection magnitude [m]")
        plt.legend()
        plt.tight_layout()
        plt.savefig("hw2_load_sweep.png", dpi=180)
        plt.close()
        print("Saved hw2_load_sweep.png")


if __name__ == "__main__":
    main()
