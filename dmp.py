import numpy as np
from math import sin, log


class DmpBase(object):
    def __init__(self, tau, x0, xd0, g):
        self.tau = tau
        self.alpha = 25.0
        self.beta = 6.25
        self.g = g
        self.x = x0
        self.xd = xd0
        self.xdd = np.zeros_like(self.x)
        self.t = 0.0
        self.f = 0.0

    def run(self, dt, t0=0.0, execution_time=None):
        """runs the whole dmp and returns ([ts], [ys], [yds])"""
        ts = []
        xs = []
        xds = []
        fs = []
        t = t0
        while t < execution_time:
            ts.append(t)
            xs.append(self.x)
            xds.append(self.xd)
            fs.append(self.f)
            t += dt
            self.step(dt)
        ts.append(t)
        xs.append(self.x)
        xds.append(self.xd)
        fs.append(self.f)

        return ts, xs, xds, fs

    def step(self, dt):
        self.t += dt
        self.x += self.xd * dt
        self.xd += self.xdd * dt
        self.xdd = self.transformation_system()
        self.f = self.forcing_term()
        self.xdd = self.transformation_system() + self.f

    def forcing_term(self, phase=None):
        return 0.0

    def transformation_system(self):
        return 0.0


class SimpleDmp(DmpBase):
    """A simple Ijspeert DMP without forcing term."""
    def transformation_system(self):
        return (self.alpha * (self.beta * (self.g - self.x) - self.tau * self.xd)) / self.tau ** 2


class SinDmp(SimpleDmp):
    """A simple Ijspeert DMP with sin(t) forcing term."""
    def __init__(self, tau, x0, xd0, g, s):
        super(SinDmp, self).__init__(tau, x0, xd0, g)
        self.s = s
        self.f = sin(self.t * 10) * self.s

    def forcing_term(self, phase=None):
        return sin(self.t * 10) * self.s


class CanonicalSystem:
    def __init__(self, tau, last_s = 0.01):
        self.tau = tau
        self.alpha = -log(last_s)
        self.s = 1.0  # current phase value
        self.t = 0.0  # current time

    def step(self, dt):
        """initially the cs is at phase 1. the first call to step will move it."""
        self.s += (-self.alpha * self.s / self.tau) * dt
        self.t += dt
        return self.s

    def get_phases(self, times):
        return np.exp(-self.alpha / self.tau * times)

    def reset(self):
        self.s = 1.0
        self.t = 0.0


class SinDmpWithCS(SinDmp):
    """A simple Ijspeert dmp with sin(t) forcing term"""
    def __init__(self, tau, x0, x0d, g, s):
        super(SinDmpWithCS, self).__init__(tau, x0, x0d, g, s)
        self.cs = CanonicalSystem(tau)

    def step(self, dt):
        self.t += dt
        self.x += self.xd * dt
        self.xd += self.xdd * dt
        phase = self.cs.step(dt)
        self.f = self.forcing_term(phase)
        self.xdd = self.transformation_system() + self.f

    def forcing_term(self, phase):
        return sin(self.t * 10) * self.s * phase


class SinDmpWithCS2(SinDmpWithCS):
    """A simple Ijspeert dmp with sin(z) forcing term"""
    def __init__(self, tau, x0, x0d, g, s):
        super(SinDmpWithCS2, self).__init__(tau, x0, x0d, g, s)

    def step(self, dt):
        self.t += dt
        self.x += self.xd * dt
        self.xd += self.xdd * dt
        phase = self.cs.step(dt)
        self.f = self.forcing_term(phase)
        self.xdd = self.transformation_system() + self.f

    def forcing_term(self, phase):
        return sin((1.0 - phase) * 10) * self.s * phase


class Rbf:
    """A simple radial basis function approximator"""
    def __init__(self, cs, executionTime, numWeights = 30, overlap = 0.1):
        """The cs is only needed to spread the centers.

        ExecutionTime is only needed to spread the centers. needs to be the
        original executionTime the the weights have been learned with
        """
        self.numWeights = numWeights
        self.cs = cs
        self.executionTime = executionTime
        #evenly spread the centers throughout the execution time
        centers_in_time = np.linspace(start=0.0, stop=executionTime, num=numWeights)
        #and move them to phase space
        self.centers = np.exp(-cs.alpha / executionTime * centers_in_time)
        self.weights = np.zeros_like(self.centers)

        #set the widths of each rbf according to overlap
        self.widths = np.ndarray(numWeights)
        log_overlap = -log(overlap)
        for i in range(1, numWeights):
            self.widths[i - 1] = log_overlap / ((self.centers[i] - self.centers[i - 1])**2)
        self.widths[numWeights - 1] = self.widths[numWeights - 2];

    def set_weights(self, weights):
        assert(len(weights) == self.numWeights)
        self.weights = weights

    def evaluate(self, z):
        psi = np.exp(-self.widths * (z - self.centers)**2)
        nom = np.dot(self.weights, psi) * z
        denom = np.sum(psi)
        return nom / denom

    def psi(self, i, phases):
        """evaluates the i'th gaussian at the specified phases and returns the results as vector"""
        assert(i < self.numWeights)
        assert(i >= 0)
        return np.exp(-self.widths[i] * (phases - self.centers[i])**2)


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
        self.z = self.tau * x0d
        self.startZ = self.z
        self.rbf = Rbf(cs, tau, n_weights, overlap)
        self.amplitude = 0
        self.scale = scale

    def step(self, dt):
        phase = self.cs.step(dt)
        f = self.rbf.evaluate(phase)
        if self.scale:
            f *= (self.g - self.x0) / self.amplitude

        zd = ((self.alpha * (self.beta * (self.g - self.x) - self.z) + f) / self.tau) * dt
        yd = self.z / self.tau * dt
        self.x += yd
        self.z += zd

    def imitate(self, times, positions, dt):
        """first position at t=0 and last position at t = executionTime
        dt = sampling dt"""
        self.amplitude = positions[-1] - positions[0]
        velocities = np.gradient(positions, dt)
        accelerations = np.gradient(velocities, dt)
        goal = positions[-1]

        forces = self.tau ** 2 * accelerations - self.alpha * (self.beta * (goal - positions) - self.tau * velocities)
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
            xds.append(self.z / self.tau)
            t += dt
            self.step(dt)
        ts.append(t)
        xs.append(self.x)
        xds.append(self.z / self.tau)
        return ts, xs, xds

    def reset(self, cs, goal, tau, x0):
        self.cs = cs
        self.g = goal
        self.tau = tau
        self.x = x0
        self.z = self.startZ