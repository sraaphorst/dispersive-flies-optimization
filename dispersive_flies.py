#!/usr/bin/env python3
#
# By Sebastian Raaphorst, 2018.
#
# An implementation of the Dispersive Flies Optimization algorithm, which can be found here:
# https://annals-csis.org/Volume_2/pliks/142.pdf
#
# This implementation is, by default, oriented towards discrete-valued problems, but passing parameters makes it
# easily usable with continuous values.

import numpy as np


def discrete_clamper(dmin, dmax, fly):
    """
    A value adjuster that constrains the value to the dmin and dmax given, and then picks the closest discrete
    value in the range.
    """
    return np.maximum(dmin, np.minimum(dmax, np.round(fly)))


def identity(_1, _2, fly):
    return fly


def manhattan_metric(f1, f2):
    """
    The Manhattan metric for fly distances.
    """
    return np.linalg.norm(f1 - f2, ord=1)


def euclidean_metric(f1, f2):
    """
    The Euclidean metric for fly distances.
    """
    return np.linalg.norm(f1 - f2)


class DispersiveFlies:
    """
    An implementation of the Dispersive Flies Optimization algorithm.
    It requires very few parameters, namely:
    1. A dimension D for the space in which the flies inhabit.
    2. A fly fitness evaluation function, which should operate on a 1D array numpy array of length D.
    3. Stopping value: stops if this value is achieved, assuming that it is the best fitness
       Default is None, meaning never stop and simply complete all rounds.
    4. The disturbance threshold, i.e. the probability a fly's coordinate is reset.
       Default is 0.025. TODO: Play with this.
    Note: values in the array are constrained to [min,max): this is why we use 2 instead of 1 for max.
    5. A numpy array to provide minimum values for each dimension.
       Default: all dimensions have minimum 0.
    6. A numpy array to provide maximum values for each dimension.
       Default: all dimensions are 2.
    7. True if the solution space is discrete, and false otherwise.
       Default: True
    8. The distance norm for finding neighbour flies.
       Default: manhattan_metric
    9. An adjustment function for values in each dimension.
       Default is discrete_clamper.
       Use the identity if you do not want any adjustment.
    10. The number of flies. Default is 500.
    11. The number of ticks (time units / rounds) to try. Default is 300,000.
    """

    class Statistics:
        def __init__(self):
            self._rounds = 0

    def __init__(self,
                 dimensions,
                 fitness,
                 stop_value=None,
                 disturbance_threshold=0.025,
                 dim_min=None,
                 dim_max=None,
                 discrete=True,
                 metric=manhattan_metric,
                 adjustment=discrete_clamper,
                 flies=500,
                 max_ticks=300000):
        self._dimensions = dimensions
        self._fitness = fitness
        self._stop_value = stop_value

        self._disturbance_threshold = disturbance_threshold

        self._dim_min = np.zeros(dimensions) if dim_min is None else dim_min
        self._dim_max = (2 * np.ones(dimensions)) if dim_max is None else dim_max
        self._discrete = discrete
        self._metric = metric

        self._adjustment = adjustment
        self._flies = flies
        self._max_ticks = max_ticks

    def _produce_random_dimension(self, d):
        value = np.random.random() * (self._dim_max[d] - self._dim_min[d]) + self._dim_min[d]
        return int(value) if self._discrete else value

    def _produce_random_fly(self):
        fly = np.random.rand(self._dimensions) * (self._dim_max - self._dim_min) + self._dim_min
        return fly.astype(int) if self._discrete else fly

    def run(self):
        s = DispersiveFlies.Statistics()

        # Create the flies randomly.
        flies = np.array([self._produce_random_fly() for _ in range(self._flies)])

        for t in range(self._max_ticks):
            # Shuffle the flies to randomize tie-breaking.
            np.random.shuffle(flies)

            # Evaluate the flies and find the best.
            evaluations = self._fitness(flies)
            best_fly = np.argmax(evaluations)

            # If there is a stop value and this fly has reached it, we are done.
            if self._stop_value and evaluations[best_fly] >= self._stop_value:
                s.ticks = t
                return s, flies[best_fly]

            # Initialize the array of new fly positions.
            new_flies = np.zeros((self._flies, self._dimensions))

            # Create one new fly at a time.
            for f in range(self._flies):
                # First, find the nearest neighbour fly. Skip over this fly, obviously, and adjust appropriately
                # if the nearest neighbour is after this fly.
                nbr = np.argmin([self._metric(flies[f], fp) for n, fp in enumerate(flies) if n != f])
                if nbr >= f:
                    nbr += 1

                for d in range(self._dimensions):
                    # Calculation is as follows:
                    # Nearest nbr + U(0,1) * (best fly - this fly)
                    new_fly = flies[nbr] + np.random.rand(self._dimensions) * (flies[best_fly] - flies[f])

                    # For each coordinate, we check if we replace it randomly.
                    # We use a list comprehension instead of np.vectorize due to problems with vectorize.
                    new_fly_adj = np.array([self._produce_random_dimension(d)
                                            if np.random.random() < self._disturbance_threshold
                                            else value
                                            for d, value in enumerate(new_fly)])
                    new_flies[f] = self._adjustment(self._dim_min, self._dim_max, new_fly_adj)

        # If we reach this point, we have run out of ticks.
        # Get and return the best fly.
        best_fly = np.argmax(self._fitness(flies))
        s.ticks = self._max_ticks
        return s, flies[best_fly]
