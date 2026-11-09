"""
MAE 263F - Homework 3
From a straight beam to a naturally curved rod (Discrete Elastic Rods, DER)

Run with a single command:   python Homework3.py

The script
  * defines run(...): the DER simulator of main_discrete_elastic_rods.py written as a function,
  * answers Questions 1-11 (prints the numbers that the multiple-choice answers are based on),
  * saves a few figures (PNG files) in the current folder.

Helper files (all in this folder): der_functions.py, computekappa.py, crossMat.py,
rotateAxisAngle.py, gradEs_hessEs.py, gradEb_hessEb.py, gradEt_hessEt.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no pop-up windows: figures are saved to files
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from der_functions import (computeTangent, assignOrthonormalAdaptedFrame, computeSpaceParallel,
                           computeMaterialDirectors, getKappa,
                           computeTimeParallel, getRefTwist, getFs, getFb, getFt)

# Default parameters (Part 1)
L_DEFAULT = 0.2        # free length [m]
R0 = 1e-3              # cross-sectional radius [m]
E_DEFAULT = 10e6       # Young's modulus [Pa]
NU = 0.5               # Poisson's ratio
RHO = 1000.0           # density [kg/m^3]
G_DEFAULT = 9.81       # gravity [m/s^2]


# ---------------------------------------------------------------------------
# Objective function: one implicit Euler step solved with Newton's method
# (same as in main_discrete_elastic_rods.py, without printing)
# ---------------------------------------------------------------------------
def objfun(qGuess, q0, u, a1, a2, freeIndex, dt, tol, refTwist, massVector, mMat,
           EA, refLen, EI, GJ, voronoiRefLen, kappaBar, twistBar, Fg, maxIter=200):
    q = qGuess
    it = 0
    error = 10 * tol

    while error > tol and it < maxIter:
        a1Iterate, a2Iterate = computeTimeParallel(a1, q0, q)
        tangent = computeTangent(q)
        refTwist_iterate = getRefTwist(a1Iterate, tangent, refTwist)
        theta = q[3::4]
        m1, m2 = computeMaterialDirectors(a1Iterate, a2Iterate, theta)

        Fb, Jb = getFb(q, m1, m2, kappaBar, EI, voronoiRefLen)
        Ft, Jt = getFt(q, refTwist_iterate, twistBar, GJ, voronoiRefLen)
        Fs, Js = getFs(q, EA, refLen)

        Forces = Fb + Ft + Fs + Fg
        JForces = Jb + Jt + Js
        f = np.dot(mMat / dt, (q - q0) / dt - u) - Forces
        J = mMat / dt**2 - JForces
        f_free = f[freeIndex]
        J_free = J[np.ix_(freeIndex, freeIndex)]
        dq_free = np.linalg.solve(J_free, f_free)

        q[freeIndex] = q[freeIndex] - dq_free
        error = np.sum(np.abs(f_free))
        it += 1

    u = (q - q0) / dt
    a1 = a1Iterate
    a2 = a2Iterate
    return q, u, a1, a2


# ---------------------------------------------------------------------------
# DER simulation as a function
#
#   N          : number of nodes (nv)
#   L          : straight rod -> free length (last N-2 edges), Eq. (1);
#                curved rod   -> arc length of the whole rod (all N-1 edges)
#   E          : Young's modulus
#   g          : gravitational acceleration magnitude (acts along -z)
#   natR       : natural radius (0 -> straight rod, >0 -> circular arc)
#   dt         : time step
#   totalTime  : total simulation time
#   removeNatCurv : if True, set kappaBar = 0 (Question 11)
#
# Returns a dictionary with the final DOF vector q and the histories of
# tip z-coordinate, tip distance ||xN - x1||, and twist angles.
# ---------------------------------------------------------------------------
def run(N=52, L=L_DEFAULT, E=E_DEFAULT, g=G_DEFAULT, natR=0.0, dt=0.1, totalTime=10.0,
        removeNatCurv=False):
    nv = N
    ne = nv - 1
    ndof = 3 * nv + ne

    # --- Nodes -------------------------------------------------------------
    nodes = np.zeros((nv, 3))
    if natR == 0:
        # Straight clamped-free rod, Eq. (1): free length L = last nv-2 edges
        dl = L / (nv - 2)
        for c in range(nv):
            nodes[c, 0] = (c - 1) * dl  # nodes[0] at x = -dl, nodes[1] at x = 0
    else:
        # Circular arc of length L (all ne edges) with natural radius natR
        dTheta = (L / natR) * (1 / ne)
        for c in range(nv):
            nodes[c, 0] = natR * np.cos(c * dTheta)
            nodes[c, 1] = natR * np.sin(c * dTheta)
            nodes[c, 2] = 0

    # --- Stiffnesses -------------------------------------------------------
    G = E / (2 * (1 + NU))
    EI = E * np.pi * R0**4 / 4
    GJ = G * np.pi * R0**4 / 2
    EA = E * np.pi * R0**2

    # --- Reference lengths -------------------------------------------------
    refLen = np.zeros(ne)
    for c in range(ne):
        refLen[c] = np.linalg.norm(nodes[c + 1, :] - nodes[c, :])

    voronoiRefLen = np.zeros(nv)
    for c in range(nv):
        if c == 0:
            voronoiRefLen[c] = 0.5 * refLen[c]
        elif c == nv - 1:
            voronoiRefLen[c] = 0.5 * refLen[c - 1]
        else:
            voronoiRefLen[c] = 0.5 * (refLen[c - 1] + refLen[c])

    # --- Mass (mass per edge = rho*A*edge length, using the TOTAL rod length) ---
    totalLength = np.sum(refLen)
    totalM = np.pi * R0**2 * totalLength * RHO
    dm = totalM / ne
    massVector = np.zeros(ndof)
    for c in range(nv):
        ind = [4 * c, 4 * c + 1, 4 * c + 2]
        if c == 0 or c == nv - 1:
            massVector[ind] = dm / 2
        else:
            massVector[ind] = dm
    for c in range(ne):
        massVector[4 * c + 3] = 0.5 * dm * R0**2
    mMat = np.diag(massVector)

    # --- Gravity -----------------------------------------------------------
    gvec = np.array([0, 0, -g])
    Fg = np.zeros(ndof)
    for c in range(nv):
        ind = [4 * c, 4 * c + 1, 4 * c + 2]
        Fg[ind] = massVector[ind] * gvec

    # --- Initial DOF vector, velocity --------------------------------------
    q0 = np.zeros(ndof)
    for c in range(nv):
        q0[4 * c:4 * c + 3] = nodes[c, :]
    u = np.zeros_like(q0)

    # --- Frames, natural curvature/twist -----------------------------------
    tangent = computeTangent(q0)
    a1_first, a2_first = assignOrthonormalAdaptedFrame(tangent[0, :])
    a1, a2 = computeSpaceParallel(a1_first, q0)
    theta = q0[3::4]
    m1, m2 = computeMaterialDirectors(a1, a2, theta)

    refTwist = np.zeros(nv)
    kappaBar = getKappa(q0, m1, m2)
    if removeNatCurv:
        kappaBar = np.zeros_like(kappaBar)  # remove the natural curvature (Question 11)
    twistBar = np.zeros(nv)

    # --- Boundary conditions: first 7 DOFs fixed (x1, theta1, x2) ----------
    fixedIndex = np.arange(0, 7)
    freeIndex = np.arange(7, ndof)

    # --- Tolerance ---------------------------------------------------------
    # Same as in class (EI/L^2*1e-3), but scaled with the load so that it is also meaningful
    # for very small gravity (the class tolerance would be larger than the gravity force
    # itself for g = 0.05 m/s^2).
    weightPerNode = dm * g
    tol = EI / L**2 * 1e-3
    if g > 0:
        tol = min(tol, 1e-4 * weightPerNode * nv)

    # --- Time stepping -----------------------------------------------------
    Nsteps = round(totalTime / dt)
    tipZ = np.zeros(Nsteps)
    tipDist = np.zeros(Nsteps)
    thetaHist = np.zeros((Nsteps, ne))
    q = q0.copy()
    for timeStep in range(Nsteps):
        qGuess = q0.copy()
        q, u, a1, a2 = objfun(qGuess, q0, u, a1, a2, freeIndex, dt, tol, refTwist,
                              massVector, mMat, EA, refLen, EI, GJ, voronoiRefLen,
                              kappaBar, twistBar, Fg)
        q0 = q.copy()
        tipZ[timeStep] = q[-1]
        tipDist[timeStep] = np.linalg.norm(q[-3:] - q[0:3])
        thetaHist[timeStep, :] = q[3::4]

    return {"q": q, "tipZ": tipZ, "tipDist": tipDist, "thetaHist": thetaHist,
            "dt": dt, "N": N, "L": L, "E": E, "g": g}


# ---------------------------------------------------------------------------
# Helpers: beam theory, steady-state check
# ---------------------------------------------------------------------------
def deltaEB(L=L_DEFAULT, E=E_DEFAULT, g=G_DEFAULT):
    """Euler-Bernoulli tip deflection of a cantilever under its own weight, Eq. (2)."""
    A = np.pi * R0**2
    I = np.pi * R0**4 / 4
    qw = RHO * A * g
    return qw * L**4 / (8 * E * I)


def isSteady(history, dt, tolRel=1e-3):
    """True if the monitored quantity changed by less than tolRel (relative) over the last second."""
    n = max(1, int(round(1.0 / dt)))
    return abs(history[-1] - history[-1 - n]) < tolRel * abs(history[-1]) if abs(history[-1]) > 0 else True


def tipDeflection(res):
    return -res["q"][-1]


# ---------------------------------------------------------------------------
# Main: answer every question
# ---------------------------------------------------------------------------
def main():
    dt, T = 0.1, 10.0
    mm = 1e3

    print("=" * 70)
    print("PART 1: straight clamped-free rod")
    print("=" * 70)

    # --- Q1: beam theory, g = 0.05
    dEB_q1 = deltaEB(g=0.05)
    print(f"\nQ1: delta_EB (g = 0.05) = {dEB_q1 * mm:.4f} mm")

    # --- Q2: DER, g = 0.05
    res = run(N=52, L=0.2, E=10e6, g=0.05, natR=0, dt=dt, totalTime=T)
    dDER_q2 = tipDeflection(res)
    print(f"Q2: delta_DER (g = 0.05, N = 52) = {dDER_q2 * mm:.4f} mm   "
          f"(steady: {isSteady(res['tipZ'], dt)}, e = {abs(dDER_q2 - dEB_q1) / dEB_q1 * 100:.2f} %)")

    # --- Q3: conceptual (see README / answer list)

    # --- Q4: mesh convergence
    print("\nQ4: mesh convergence (g = 0.05, L = 0.2 m)")
    Ns = [22, 42, 82]
    dDER_N = []
    for N in Ns:
        r = run(N=N, L=0.2, E=10e6, g=0.05, natR=0, dt=dt, totalTime=T)
        d = tipDeflection(r)
        dDER_N.append(d)
        print(f"   N = {N:3d}: delta_DER = {d * mm:.4f} mm   (delta_DER/delta_EB = {d / dEB_q1:.4f}, "
              f"diff = {(d - dEB_q1) * mm:+.4f} mm, steady: {isSteady(r['tipZ'], dt)})")

    # --- Q5: stiffer rods, g = 9.81
    print("\nQ5: stiffer rods (g = 9.81)")
    E_list = [10e6, 30e6, 100e6, 300e6, 1000e6]
    e_E = []
    zs = []
    for E in E_list:
        r = run(N=52, L=0.2, E=E, g=9.81, natR=0, dt=dt, totalTime=T)
        d = tipDeflection(r)
        dEB = deltaEB(E=E, g=9.81)
        e = abs(d - dEB) / dEB
        e_E.append(e)
        zs.append((d, dEB))
        print(f"   E = {E / 1e6:6.0f} MPa: delta_DER = {d * mm:9.4f} mm, delta_EB = {dEB * mm:9.4f} mm, "
              f"e = {e * 100:7.2f} %  (steady: {isSteady(r['tipZ'], dt)})")

    # --- Q6: heavier load, E = 10 MPa
    print("\nQ6: heavier load (E = 10 MPa)")
    g_list = [0.1, 0.3, 1.0, 3.0, 9.81]
    e_g = []
    for gval in g_list:
        r = run(N=52, L=0.2, E=10e6, g=gval, natR=0, dt=dt, totalTime=T)
        d = tipDeflection(r)
        dEB = deltaEB(g=gval)
        e = abs(d - dEB) / dEB
        e_g.append(e)
        print(f"   g = {gval:5.2f}: delta_DER = {d * mm:9.4f} mm, delta_EB = {dEB * mm:9.4f} mm, "
              f"e = {e * 100:7.2f} %  (steady: {isSteady(r['tipZ'], dt)})")

    # --- Q7: default parameters (E = 10 MPa, g = 9.81)
    resStraight = run(N=52, L=0.2, E=10e6, g=9.81, natR=0, dt=dt, totalTime=T)
    d7 = tipDeflection(resStraight)
    print(f"\nQ7: delta_DER (E = 10 MPa, g = 9.81) = {d7:.4f} m = {d7 * mm:.1f} mm   "
          f"(delta_EB = {deltaEB() :.4f} m, steady: {isSteady(resStraight['tipZ'], dt)})")
    thetaStraight = resStraight["q"][3::4]

    # --- Q8: conceptual (see answers)

    print("\n" + "=" * 70)
    print("PART 2: naturally curved rod (N = 50, dt = 0.1, T = 10, g = 9.81)")
    print("=" * 70)

    # --- Q9: curved rod
    resCurved = run(N=50, L=0.2, E=10e6, g=9.81, natR=0.02, dt=dt, totalTime=T)
    zN = resCurved["q"][-1]
    print(f"\nQ9: z-coordinate of last node = {zN:.4f} m  (steady: {isSteady(resCurved['tipZ'], dt)})")

    # --- Q10: twist angles
    thetaCurved = resCurved["q"][3::4]
    print("\nQ10: twist angles theta_k = q[3::4]")
    print(f"   straight rod (Q7): max |theta| = {np.max(np.abs(thetaStraight)):.3e} rad")
    print(f"   curved rod   (Q9): max |theta| = {np.max(np.abs(thetaCurved)):.3e} rad "
          f"({np.degrees(np.max(np.abs(thetaCurved))):.2f} deg)")
    thetaChange = np.max(np.abs(resCurved["thetaHist"][-1] - resCurved["thetaHist"][-1 - int(1 / dt)]))
    print(f"   curved rod: max change of theta over the last second = {thetaChange:.3e} rad")

    # --- Q11: no natural curvature, g = 0
    resFlat = run(N=50, L=0.2, E=10e6, g=0.0, natR=0.02, dt=dt, totalTime=T, removeNatCurv=True)
    dist0 = np.linalg.norm(np.array([0.02 * np.cos(49 * (0.2 / 0.02) / 49), 0.02 * np.sin(49 * (0.2 / 0.02) / 49), 0.0])
                           - np.array([0.02, 0.0, 0.0]))
    distF = resFlat["tipDist"][-1]
    print(f"\nQ11: ||xN - x1|| initial = {dist0:.4f} m, final = {distF:.4f} m "
          f"(change over last second: {abs(resFlat['tipDist'][-1] - resFlat['tipDist'][-1 - int(1 / dt)]):.2e} m)")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    # Mesh convergence
    plt.figure()
    plt.plot([22, 42, 82], np.array(dDER_N) * mm, 'bo-', label='DER')
    plt.axhline(dEB_q1 * mm, color='r', ls='--', label='Euler-Bernoulli')
    plt.xlabel('N (number of nodes)')
    plt.ylabel('Tip deflection [mm]')
    plt.title('Q4: mesh convergence, g = 0.05 m/s$^2$')
    plt.legend()
    plt.savefig('fig_q4_mesh_convergence.png', dpi=150)

    # Relative difference vs E and g
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(np.array(E_list) / 1e6, np.array(e_E) * 100, 'bo-')
    ax[0].axhline(20, color='r', ls='--')
    ax[0].set_xlabel('E [MPa]')
    ax[0].set_ylabel('e [%]')
    ax[0].set_title('Q5 (g = 9.81)')
    ax[1].loglog(g_list, np.array(e_g) * 100, 'bo-')
    ax[1].axhline(20, color='r', ls='--')
    ax[1].set_xlabel('g [m/s$^2$]')
    ax[1].set_ylabel('e [%]')
    ax[1].set_title('Q6 (E = 10 MPa)')
    plt.tight_layout()
    plt.savefig('fig_q5_q6_relative_difference.png', dpi=150)

    # Rod shapes
    fig = plt.figure(figsize=(12, 4))
    for k, (name, rr) in enumerate([("Straight rod (Q7)", resStraight),
                                    ("Curved rod (Q9)", resCurved),
                                    ("No natural curvature, g = 0 (Q11)", resFlat)]):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        qq = rr["q"]
        ax.plot3D(qq[0::4], qq[1::4], qq[2::4], 'ko-', ms=2)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.tight_layout()
    plt.savefig('fig_rod_shapes.png', dpi=150)

    # Tip time histories
    plt.figure()
    t = np.arange(1, len(resStraight["tipZ"]) + 1) * dt
    plt.plot(t, resStraight["tipZ"], 'b-', label='straight rod (Q7)')
    plt.plot(t, resCurved["tipZ"], 'r-', label='curved rod (Q9)')
    plt.xlabel('Time [s]')
    plt.ylabel('z-coordinate of last node [m]')
    plt.legend()
    plt.savefig('fig_tip_z_vs_time.png', dpi=150)

    print("\nFigures saved in the current folder (fig_*.png).")


if __name__ == "__main__":
    main()
