import numpy as np
import matplotlib.pyplot as plt


class CanonicalSystem:
    def __init__(self, tau, last_s=0.01):
        self.tau = tau
        self.alpha = -np.log(last_s)
        self.z = 1.0  # current phase value

    def step(self, dt):
        """initially the cs is at phase 1. the first call to step will move it."""
        self.z += (-self.alpha * self.z / self.tau) * dt
        return self.z

    def get_phases(self, times):
        return np.exp(-self.alpha / self.tau * times)

    def reset(self):
        self.z = 1.0


class Rbf:
    """A simple radial basis function approximator"""
    def __init__(self, cs, tau, n_weights=30, overlap=0.1):
        """The cs is only needed to spread the centers.

        ExecutionTime is only needed to spread the centers. needs to be the
        original executionTime the the weights have been learned with
        """
        self.numWeights = n_weights
        self.cs = cs
        self.executionTime = tau
        # evenly spread the centers throughout the execution time
        centers_in_time = np.linspace(0.0, tau, n_weights)
        # and move them to phase space
        self.centers = np.exp(-cs.alpha / tau * centers_in_time)
        self.weights = np.zeros_like(self.centers)

        # set the widths of each rbf according to overlap
        self.widths = np.ndarray(n_weights)
        log_overlap = -np.log(overlap)
        for i in range(1, n_weights):
            self.widths[i - 1] = log_overlap / (
            (self.centers[i] - self.centers[i - 1]) ** 2)
        self.widths[n_weights - 1] = self.widths[n_weights - 2]

    def set_weights(self, weights):
        assert (len(weights) == self.numWeights)
        self.weights = weights

    def evaluate(self, z):
        psi = np.exp(-self.widths * (z - self.centers) ** 2)
        nom = np.dot(self.weights, psi) * z
        denom = np.sum(psi)
        return nom / denom

    def psi(self, i, phases):
        """evaluates the i'th gaussian at the specified phases and returns the results as vector"""
        assert (i < self.numWeights)
        assert (i >= 0)
        return np.exp(-self.widths[i] * (phases - self.centers[i]) ** 2)


class DmpWithImitation:
    """A simple Ijspeert dmp with forcing term"""
    def __init__(self, tau, x0, x0d, g, cs, n_weights, overlap, scale):
        self.tau = tau
        self.cs = cs
        self.alpha = 25.0
        self.beta = 6.25
        self.g = g
        self.x = x0
        self.x0 = x0
        self.xd = x0d
        self.start_xd = self.xd
        self.rbf = Rbf(cs, tau, n_weights, overlap)
        self.amplitude = 0
        self.scale = scale

    def step(self, dt):
        phase = self.cs.step(dt)
        f = self.rbf.evaluate(phase)
        if self.scale:
            f *= (self.g - self.x0) / self.amplitude

        xdd = (self.alpha * (self.beta * (self.g - self.x) - self.xd) + f) / self.tau ** 2
        self.x += self.xd * dt
        self.xd += xdd * dt

    def imitate(self, times, positions, dt):
        """first position at t=0 and last position at t = executionTime
        dt = sampling dt"""
        self.amplitude = positions[-1] - positions[0]
        velocities = np.gradient(positions, dt)
        accelerations = np.gradient(velocities, dt)
        goal = positions[-1]

        forces = self.tau ** 2 * accelerations - self.alpha * (
            self.beta * (goal - positions) - self.tau * velocities)
        phases = self.cs.get_phases(times)

        weights = np.ndarray(self.rbf.numWeights)
        for i in range(self.rbf.numWeights):
            psi = self.rbf.psi(i, phases)
            psiD = np.diag(psi)
            weights[i] = (np.linalg.inv([[np.dot(phases.T, np.dot(psiD, phases))]]) *
                          np.dot(phases.T, np.dot(psiD, forces)))
        self.rbf.set_weights(weights)

    def run(self, dt, t0=0.0, execution_time=None):
        ts = []
        xs = []
        xds = []
        t = t0
        if execution_time is None:
            execution_time = self.tau
        while t < execution_time:
            ts.append(t)
            xs.append(self.x)
            xds.append(self.xd / self.tau)
            t += dt
            self.step(dt)
        ts.append(t)
        xs.append(self.x)
        xds.append(self.xd / self.tau)
        return ts, xs, xds

    def reset(self, cs, goal, tau, x0):
        self.cs = cs
        self.g = goal
        self.tau = tau
        self.x = x0
        self.x0 = x0
        self.xd = self.start_xd


def plot_imitation(demo, ax=None, legend=True):
    dt = 0.01
    tau = dt * len(demo)

    times = np.linspace(0.0, tau, len(demo))

    cs = CanonicalSystem(tau)
    dmp = DmpWithImitation(
        tau, demo[0], (demo[1] - demo[0]) / dt, demo[-1], cs, n_weights=10,
        overlap=0.2, scale=False)
    dmp.imitate(times, demo, dt)

    if ax is None:
        ax = plt.subplot(111)
    ax.plot(times, demo, "r")

    dmp.reset(CanonicalSystem(tau), demo[-1], tau, demo[0])
    ts, ys, yds = dmp.run(0.001)
    ax.plot(ts, ys)

    dmp.reset(CanonicalSystem(tau), -0.5, tau, 0)
    ts, ys, yds = dmp.run(0.001)
    ax.plot(ts, ys)

    if legend:
        ax.legend(["Demo", "Imitated", "New start and goal"], loc="upper left")

    return ax


demo = np.sin(np.linspace(0, 1.5 * np.pi, 100))
ax = plot_imitation(demo)
ax.set_xlabel("Time")
ax.set_ylabel("X Position")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-2.0, 2.0)
plt.show()
