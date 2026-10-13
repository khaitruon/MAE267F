"""
MAE-263F Homework 1: Modular multi-mass-spring-damper simulator.
Run with:  python Homework1.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt

G = 9.8


# ----------------------------------------------------------------------------
# Element functions
# ----------------------------------------------------------------------------
def gradEs(xa, xb, k):
    """Gradient of spring energy wrt (xa, xb)."""
    return k * (xb - xa) * np.array([-1.0, 1.0])


def hessEs(k):
    """Hessian of spring energy."""
    return k * np.array([[1.0, -1.0], [-1.0, 1.0]])


def dampingForce(xa_new, xb_new, xa_old, xb_old, b, dt):
    """Damper force vector g_i with velocities (x_new - x_old)/dt."""
    ua = (xa_new - xa_old) / dt
    ub = (xb_new - xb_old) / dt
    return b * (ub - ua) * np.array([-1.0, 1.0])


def dampingJacobian(b, dt):
    """Derivative of dampingForce wrt (xa_new, xb_new)."""
    return (b / dt) * np.array([[1.0, -1.0], [-1.0, 1.0]])


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------
def getFJ(x_new, x_old, u_old, params):
    m, k, b, g, dt = params['m'], params['k'], params['b'], params['g'], params['dt']
    N = len(x_new) - 1
    f = np.zeros(N + 1)
    J = np.zeros((N + 1, N + 1))

    # inertia + gravity (nodes 0..N; row 0 is removed later)
    f += m / dt * ((x_new - x_old) / dt - u_old) - m * g
    J += np.diag(m) / dt**2

    for i in range(1, N + 1):
        idx = [i - 1, i]
        # spring
        f[idx] += gradEs(x_new[i - 1], x_new[i], k[i])
        J[np.ix_(idx, idx)] += hessEs(k[i])
        # damper
        f[idx] += dampingForce(x_new[i - 1], x_new[i], x_old[i - 1], x_old[i], b[i], dt)
        J[np.ix_(idx, idx)] += dampingJacobian(b[i], dt)
    return f, J


# ----------------------------------------------------------------------------
# Time steppers
# ----------------------------------------------------------------------------
def implicitEulerStep(x_old, u_old, params, eps=1e-10, max_iter=50, return_iters=False):
    x_new = x_old.copy()
    N = len(x_old) - 1
    free = np.arange(1, N + 1)
    err = 10 * eps
    it = 0
    while err > eps and it < max_iter:
        f, J = getFJ(x_new, x_old, u_old, params)
        dx = np.linalg.solve(J[np.ix_(free, free)], f[free])
        x_new[free] -= dx
        f, _ = getFJ(x_new, x_old, u_old, params)
        err = np.linalg.norm(f[free])
        it += 1
    u_new = (x_new - x_old) / params['dt']
    if return_iters:
        return x_new, u_new, it, err
    return x_new, u_new


def explicitEulerStep(x_old, u_old, params):
    m, k, b, g, dt = params['m'], params['k'], params['b'], params['g'], params['dt']
    N = len(x_old) - 1
    force = np.zeros(N + 1)  # sum of spring + damper forces (gradient terms)
    for i in range(1, N + 1):
        idx = [i - 1, i]
        force[idx] += gradEs(x_old[i - 1], x_old[i], k[i])
        force[idx] += b[i] * (u_old[i] - u_old[i - 1]) * np.array([-1.0, 1.0])
    a = np.zeros(N + 1)
    a[1:] = g - force[1:] / m[1:]
    x_new = x_old + dt * u_old
    u_new = u_old + dt * a
    x_new[0] = 0.0
    u_new[0] = 0.0
    return x_new, u_new


# ----------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------
def simulate(m, k, b, dt, T, step=implicitEulerStep, g=G):
    """m, k, b given for i = 1..N. Returns t, x, u (arrays of shape (steps+1, N+1))."""
    N = len(m)
    params = {'m': np.concatenate(([0.0], m)).astype(float),
              'k': np.concatenate(([0.0], k)).astype(float),
              'b': np.concatenate(([0.0], b)).astype(float),
              'g': g, 'dt': dt}
    nsteps = int(round(T / dt))
    t = np.arange(nsteps + 1) * dt
    x = np.zeros((nsteps + 1, N + 1))
    u = np.zeros((nsteps + 1, N + 1))
    for n in range(nsteps):
        x[n + 1], u[n + 1] = step(x[n], u[n], params)
    return t, x, u


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def run_tests():
    assert np.allclose(gradEs(0, 0.1, 10), [-1, 1])
    assert np.allclose(hessEs(10), [[10, -10], [-10, 10]])
    assert np.allclose(dampingForce(0, 0.01, 0, 0, 2, 0.01), [-2, 2])
    assert np.allclose(dampingJacobian(2, 0.01), [[200, -200], [-200, 200]])

    # Test 5
    p = {'m': np.zeros(4), 'k': np.array([0, 1, 2, 3.]), 'b': np.zeros(4), 'g': 0.0, 'dt': 1.0}
    z = np.zeros(4)
    _, J = getFJ(z, z, z, p)
    assert np.allclose(J[1:, 1:], [[3, -2, 0], [-2, 5, -3], [0, -3, 3]])

    # Test 6
    p = {'m': np.array([0, 1, 2, 3.]), 'k': np.array([0, 10, 20, 30.]),
         'b': np.array([0, 1, 2, 3.]), 'g': 9.8, 'dt': 0.1}
    x_old = np.zeros(4)
    u_old = np.array([0, 0.1, 0.2, 0.3])
    x_new = np.array([0, 0.01, 0.03, 0.06])
    f, J = getFJ(x_new, x_old, u_old, p)
    assert np.allclose(f[1:], [-10.4, -18.6, -18.6])
    assert np.allclose(J[1:, 1:], [[160, -40, 0], [-40, 300, -60], [0, -60, 360]])

    # Test 7
    p7 = {'m': np.array([0, 1.]), 'k': np.zeros(2), 'b': np.zeros(2), 'g': 9.8, 'dt': 0.01}
    xn, un, it, err = implicitEulerStep(np.zeros(2), np.zeros(2), p7, return_iters=True)
    assert np.allclose(xn, [0, 0.00098]) and np.allclose(un, [0, 0.098])
    assert it == 1 and err < 1e-9   # Test 9

    # Test 8
    xn, un, it, err = implicitEulerStep(x_old, u_old, p, return_iters=True)
    assert np.allclose(xn, [0, 0.096696, 0.116786, 0.126131], atol=1e-6)
    assert np.allclose(un, [0, 0.966964, 1.167857, 1.261310], atol=1e-5)
    assert it == 1 and err < 1e-9   # Test 9

    # Test 10
    p10 = {'m': np.array([0, 1.]), 'k': np.array([0, 1.]), 'b': np.zeros(2), 'g': 0.0, 'dt': 0.1}
    xn, un = explicitEulerStep(np.array([0, 1.]), np.zeros(2), p10)
    assert np.allclose(xn, [0, 1]) and np.allclose(un, [0, -0.1])

    # Test 11
    xn, un = explicitEulerStep(np.array([0, 0.01, 0.03, 0.06]), np.array([0, 0.1, 0.2, 0.3]), p)
    assert np.allclose(xn, [0, 0.02, 0.05, 0.09]) and np.allclose(un, [0, 1.12, 1.21, 1.24])
    print("All 11 tests passed.\n")


# ----------------------------------------------------------------------------
# Questions
# ----------------------------------------------------------------------------
def run_questions():
    res = {}

    t, x, u = simulate([1], [0], [0], 1e-3, 1.0)
    res['Q1: x1(1 s)'] = x[-1, 1]

    t, x, u = simulate([1, 2, 3], [0, 0, 0], [0, 0, 0], 1e-3, 2.0)
    res['Q2: x3(2 s)'] = x[-1, 3]

    t, x, u = simulate([1], [0], [2], 0.01, 20.0)
    res['Q3: u1(20 s)'] = u[-1, 1]

    t, x, u = simulate([1, 1], [0, 0], [2, 2], 0.01, 20.0)
    res['Q4: u2(20 s)'] = u[-1, 2]

    t, x, u = simulate([1], [10], [1], 0.01, 50.0)
    res['Q5: x1(50 s)'] = x[-1, 1]

    t, x, u = simulate([1] * 5, [10] * 5, [5] * 5, 0.01, 100.0)
    res['Q6: x5(100 s)'] = x[-1, 5]

    t, x, u = simulate([1, 2, 3], [10, 20, 30], [5, 10, 15], 0.01, 100.0)
    res['Q7: x3(100 s)'] = x[-1, 3]

    t8, x8, _ = simulate([1], [1], [0], 0.1, 100.0)
    res['Q8: x1(100 s), dt=0.1'] = x8[-1, 1]
    amp8 = (x8[-300:, 1].max() - x8[-300:, 1].min()) / 2  # last 30 s

    t9, x9, _ = simulate([1], [1], [0], 0.01, 100.0)
    n = int(round(30 / 0.01))
    amp9 = (x9[-n:, 1].max() - x9[-n:, 1].min()) / 2
    res['Q9: x1(100 s), dt=0.01'] = x9[-1, 1]
    res['Q9: amplitude last 30 s (dt=0.1)'] = amp8
    res['Q9: amplitude last 30 s (dt=0.01)'] = amp9

    t10, x10, _ = simulate([1], [1], [0], 0.1, 100.0, step=explicitEulerStep)
    res['Q10: max |x1 - 9.8| over 100 s'] = np.max(np.abs(x10[:, 1] - 9.8))
    res['Q10: x1(100 s)'] = x10[-1, 1]

    mask = t10 <= 10.0 + 1e-9
    res['Q11: max x1 over [0, 10 s]'] = x10[mask, 1].max()

    for key, val in res.items():
        print(f"{key:40s} {val:.6g}")

    # Plots (after the loops)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t8, x8[:, 1], label='dt=0.1'); ax[0].plot(t9, x9[:, 1], label='dt=0.01')
    ax[0].set_title('Implicit Euler, undamped oscillator'); ax[0].legend()
    ax[1].plot(t10, x10[:, 1]); ax[1].set_title('Explicit Euler, dt=0.1')
    t, x, u = simulate([1, 2, 3], [10, 20, 30], [5, 10, 15], 0.01, 100.0)
    ax[2].plot(t, x[:, 1:]); ax[2].set_title('Q7 chain positions')
    for a in ax:
        a.set_xlabel('t [s]')
    plt.tight_layout()
    plt.savefig('homework1_plots.png', dpi=120)


if __name__ == "__main__":
    run_tests()
    run_questions()
