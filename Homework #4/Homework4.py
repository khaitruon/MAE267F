import numpy as np


# ---------------- Part 1: edges and hinges ----------------
def get_edges_and_hinges(elements):
    edges = {}
    for t, tri in enumerate(elements):
        for a in range(3):
            i, j = tri[a], tri[(a + 1) % 3]
            k = tri[(a + 2) % 3]
            edges.setdefault((min(i, j), max(i, j)), []).append(k)
    return edges, {e: v for e, v in edges.items() if len(v) == 2}


# ---------------- Part 2: hinge angle / energy ----------------
def getTheta(x0, x1=None, x2=None, x3=None):
    if x1 is None:                      # accept q of size 12
        q = np.asarray(x0)
        x0, x1, x2, x3 = q[0:3], q[3:6], q[6:9], q[9:12]
    n0 = np.cross(x1 - x0, x2 - x0)
    n1 = np.cross(x3 - x0, x1 - x0)
    e = x1 - x0
    e_hat = e / np.linalg.norm(e)
    n0h, n1h = n0 / np.linalg.norm(n0), n1 / np.linalg.norm(n1)
    sin_t = np.dot(np.cross(n0h, n1h), e_hat)
    cos_t = np.dot(n0h, n1h)
    return np.arctan2(sin_t, cos_t)


def Eb(q, thetaBar, kb):
    return 0.5 * kb * (getTheta(q) - thetaBar) ** 2


def gradTheta(q):
    """Analytical gradient of the hinge angle (size 12)."""
    x0, x1, x2, x3 = q[0:3], q[3:6], q[6:9], q[9:12]
    e0, e1, e2 = x1 - x0, x2 - x0, x3 - x0
    l0 = np.linalg.norm(e0)
    n1 = np.cross(e0, e1)              # normal of triangle 1
    n2 = np.cross(e2, e0)              # normal of triangle 2
    h1 = np.linalg.norm(n1) / l0       # height of x2 above the hinge
    h2 = np.linalg.norm(n2) / l0       # height of x3 above the hinge
    # theta is signed from n1 to n2, so moving x2 along +n1 decreases theta
    g2 = -n1 / np.linalg.norm(n1) / h1
    g3 = -n2 / np.linalg.norm(n2) / h2
    a1 = np.dot(e0, e1) / l0 ** 2
    a2 = np.dot(e0, e2) / l0 ** 2
    g0 = -(1 - a1) * g2 - (1 - a2) * g3
    g1 = -a1 * g2 - a2 * g3
    return np.concatenate([g0, g1, g2, g3])


def gradEb(q, thetaBar, kb):
    return kb * (getTheta(q) - thetaBar) * gradTheta(q)


if __name__ == "__main__":
    # ---- Question 1 ----
    elements = []
    for j in range(2):
        for i in range(2):
            n = 3 * j + i
            elements.append((n, n + 1, n + 4))        # lower-right triangle
            elements.append((n, n + 4, n + 3))        # upper-left triangle
    edges, hinges = get_edges_and_hinges(elements)
    print("Q1: edges =", len(edges), " hinges =", len(hinges))

    # ---- Question 2 ----
    q = np.array([0, 0, 0, 1, 0, 0, 0.5, 1, 0, 0.5, -1, 1], dtype=float)
    theta = getTheta(q)
    print("Q2: theta = %.5f rad (%.4f*pi), Eb = %.5f J" % (theta, theta / np.pi, Eb(q, 0, 1)))

    # ---- Questions 3-6 ----
    eps = 0.01
    names = ["x0", "x1", "x2", "x3"]
    E = np.zeros(12)
    for i in range(12):
        dq = np.zeros(12); dq[i] = eps
        E[i] = Eb(q + dq, 0.0, 1.0)
    for n in range(4):
        print("Q%d: %s moved -> x: %.5f  y: %.5f  z: %.5f" %
              (3 + n, names[n], E[3 * n], E[3 * n + 1], E[3 * n + 2]))

    # ---- Question 7 ----
    gradFD = (E - Eb(q, 0, 1)) / eps
    np.set_printoptions(precision=5, suppress=True)
    print("Q7: gradFD =", gradFD)

    # ---- Question 8 ----
    dF = gradEb(q, 0.0, 1.0)
    print("Q8: dF     =", dF)
