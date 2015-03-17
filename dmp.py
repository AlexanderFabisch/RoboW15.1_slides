import numpy as np
from math import sin, log


class SimpleDmp:
    """A simple Ijspeert DMP with configurable forcing term."""
    def __init__(self, executionTime, startPos, startVel, goalPos):
        self.T = executionTime
        self.alpha = 25.0
        self.beta = 6.25
        self.g = goalPos
        self.y = startPos
        self.z = self.T * startVel;

    def step(self, dt):
        f = 0.0
        zd = ((self.alpha * (self.beta * (self.g - self.y)- self.z)) / self.T) * dt
        yd = self.z / self.T * dt
        self.y += yd
        self.z += zd

    def run(self, dt, startT, endT):
        """runs the whole dmp and returns ([ts], [ys], [yds])"""
        ts = []
        ys = []
        yds = []
        t = startT
        while t < endT:
            ts.append(t)
            ys.append(self.y)
            yds.append(self.z / self.T)
            t += dt
            self.step(dt)
        ts.append(t)
        ys.append(self.y)
        yds.append(self.z / self.T)

        return (ts, ys, yds)


class SinDmp:
    """A simple Ijspeert DMP with sin(t) forcing term"""
    def __init__(self, executionTime, startPos, startVel, goalPos, s):
        self.T = executionTime
        self.alpha = 25.0
        self.beta = 6.25
        self.g = goalPos
        self.y = startPos
        self.z = self.T * startVel;
        self.t = 0.0
        self.s = s
        self.f = sin(self.t * 10) * self.s

    def step(self, dt):
        self.f = sin(self.t * 10) * self.s
        zd = ((self.alpha * (self.beta * (self.g - self.y)- self.z) + self.f) / self.T) * dt
        yd = self.z / self.T * dt
        self.y += yd
        self.z += zd
        self.t += dt

    def run(self, dt, startT, endT):
        """runs the whole dmp and returns ([ts], [ys], [yds])"""
        ts = []
        ys = []
        yds = []
        fs = []
        t = startT
        while t < endT:
            ts.append(t)
            ys.append(self.y)
            yds.append(self.z / self.T)
            fs.append(self.f)
            t += dt
            self.step(dt)
        ts.append(t)
        ys.append(self.y)
        yds.append(self.z / self.T)
        fs.append(self.f)

        return (ts, ys, yds, fs)


class CS:
    def __init__(self, executionTime, lastPhaseValue = 0.01):
        self.T = executionTime
        self.alpha = -log(lastPhaseValue)
        self.z = 1.0  # current phase value
        self.t = 0.0  # current time

    def step(self, dt):
        """initially the cs is at phase 1. the first call to step will move it."""
        self.z += (-self.alpha * self.z / self.T) * dt
        self.t += dt
        return self.z

    def get_phases(self, times):
        return np.exp(-self.alpha / self.T * times)

    def reset(self):
        self.z = 1.0
        self.t = 0.0


class SinDmpWithCS:
    """A simple Ijspeert dmp with sin(t) forcing term"""
    def __init__(self, executionTime, startPos, startVel, goalPos, s):
        self.T = executionTime
        self.alpha = 25.0
        self.beta = 6.25
        self.g = goalPos
        self.y = startPos
        self.z = self.T * startVel;
        self.t = 0.0
        self.s = s
        self.cs = CS(executionTime)

    def step(self, dt):
        f = sin(self.t * 10) * self.s
        phase = self.cs.step(dt)
        f *= phase
        zd = ((self.alpha * (self.beta * (self.g - self.y)- self.z) + f) / self.T) * dt
        yd = self.z / self.T * dt
        self.y += yd
        self.z += zd
        self.t += dt

    def run(self, dt, startT, endT):
        """runs the whole dmp and returns ([ts], [ys], [yds])"""
        ts = []
        ys = []
        yds = []
        t = startT
        while t < endT:
            ts.append(t)
            ys.append(self.y)
            yds.append(self.z / self.T)
            t += dt
            self.step(dt)
        ts.append(t)
        ys.append(self.y)
        yds.append(self.z / self.T)

        return (ts, ys, yds)


class SinDmpWithCS2:
    """A simple Ijspeert dmp with sin(z) forcing term"""
    def __init__(self, executionTime, startPos, startVel, goalPos, s):
        self.T = executionTime
        self.alpha = 25.0
        self.beta = 6.25
        self.g = goalPos
        self.y = startPos
        self.z = self.T * startVel;
        self.s = s
        self.cs = CS(executionTime)

    def step(self, dt):
        phase = self.cs.step(dt)
        f = sin((1.0 - phase) * 10) * self.s
        f *= phase
        zd = ((self.alpha * (self.beta * (self.g - self.y)- self.z) + f) / self.T) * dt
        yd = self.z / self.T * dt
        self.y += yd
        self.z += zd

    def run(self, dt, startT, endT):
        """runs the whole dmp and returns ([ts], [ys], [yds])"""
        ts = []
        ys = []
        yds = []
        t = startT
        while t < endT:
            ts.append(t)
            ys.append(self.y)
            yds.append(self.z / self.T)
            t += dt
            self.step(dt)
        ts.append(t)
        ys.append(self.y)
        yds.append(self.z / self.T)

        return (ts, ys, yds)

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

    def plot_gaussians(self):
        points_in_time = np.linspace(0.0, executionTime * 1.3, num=600)
        points_in_phase = self.cs.get_phases(points_in_time)
        for i in range(self.numWeights):#for each guassian
            values = np.exp(-self.widths[i] * (points_in_phase - self.centers[i])**2)
            plt.plot(points_in_phase, values)
            plt.plot(points_in_time, values)

    def plot_function(self):
        points_in_time = np.linspace(0.0, executionTime * 1.3, num=600)
        points_in_phase = self.cs.get_phases(points_in_time)
        values = []
        for z in points_in_phase:
            values.append(self.evaluate(z))
        plt.plot(points_in_time, values)

    def psi(self, i, phases):
        """evaluates the i'th gaussian at the specified phases and returns the results as vector"""
        assert(i < self.numWeights)
        assert(i >= 0)
        return np.exp(-self.widths[i] * (phases - self.centers[i])**2)


class DmpWithImitation:
    """A simple Ijspeert dmp with forcing term"""
    def __init__(self, executionTime, startPos, startVel, goalPos, cs,
                 numWeights, overlap, scale):
        self.T = executionTime
        self.cs = cs
        self.alpha = 25.0
        self.beta = 6.25
        self.g = goalPos
        self.y = startPos
        self.startPos = startPos
        self.z = self.T * startVel;
        self.startZ = self.z
        self.rbf = Rbf(cs, executionTime, numWeights, overlap)
        self.amplitude = 0
        self.scale = scale

    def step(self, dt):
        z = self.cs.step(dt)
        f = self.rbf.evaluate(z)
        if self.scale:
            f *= (self.g - self.startPos) / self.amplitude

        zd = ((self.alpha * (self.beta * (self.g - self.y)- self.z) + f) / self.T) * dt
        yd = self.z / self.T * dt
        self.y += yd
        self.z += zd

    def imitate(self, times, positions, dt):
        """first position at t=0 and last position at t = executionTime
        dt = sampling dt"""
        self.amplitude = positions[-1] - positions[0]
        velocities = np.gradient(positions, dt)
        accelerations = np.gradient(velocities, dt)
        goal = positions[len(positions) - 1]
        references = self.T**2 * accelerations - self.alpha * (self.beta * (goal - positions) - self.T * velocities)
        phases = self.cs.get_phases(times)
        weights = np.ndarray(self.rbf.numWeights)
        for i in range(self.rbf.numWeights):
            psi = self.rbf.psi(i, phases)
            psiD = np.diag(psi)
            weights[i] = np.linalg.inv([[np.dot(phases.T, np.dot(psiD, phases))]]) * np.dot(phases.T, np.dot(psiD, references))
        self.rbf.set_weights(weights)

    def run(self, dt, startTime = 0.0, endTime = None):
        """runs the whole dmp and returns ([ts], [ys], [yds])"""
        ts = []
        ys = []
        yds = []
        t = startTime
        if endTime is None:
            endTime = self.T
        while t < endTime:
            ts.append(t)
            ys.append(self.y)
            yds.append(self.z / self.T)
            t += dt
            self.step(dt)
        ts.append(t)
        ys.append(self.y)
        yds.append(self.z / self.T)
        return (ts, ys, yds)

    def reset(self, cs, goal, executionTime, start):
        self.cs = cs
        self.g = goal
        self.T = executionTime
        self.y = start
        self.z = self.startZ