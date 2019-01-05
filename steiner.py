#!/usr/bin/env python3
#
# By Sebastian Raaphorst, 2018.
#
# Formulation of Steiner systems to try to solve with an optimizer.

import sys
from math import factorial
from itertools import combinations
from dispersive_flies import DispersiveFlies


def steiner(t, k, v):
    # Yes, there is some redundancy here.
    # I still like to have all the conditions explicitly listed.
    if not (1 < t < k):
        raise ValueError("Must have 1 < t < k")
    if not (t < k < v):
        raise ValueError("Must have t < k < v")
    if not (v > k):
        raise ValueError("Must have v > k")

    # Check feasibility.
    vCt = factorial(v) // factorial(t) // factorial(v - t)
    kCt = factorial(k) // factorial(t) // factorial(k - t)
    if vCt % kCt != 0:
        raise ValueError("Necessary condition vCt / kCt failed")
    v1Ct1 = factorial(v - 1) // factorial(t - 1) // factorial(v - t)
    k1Ct1 = factorial(k - 1) // factorial(t - 1) // factorial(k - t)
    if v1Ct1 % k1Ct1 != 0:
        raise ValueError("Necessary condition (v-1)C(t-1) / (k-1)C(t-1) failed")

    # Formulate the problem. The number of dimensions is the number of blocks.
    total_blocks = factorial(v) // factorial(k) // factorial(v - k)

    # We need the fitness function.
    # Precalculate the list of t-sets that come from k-sets. Our score will be the number of t-sets that are
    # covered, with a score of zero if a t-set is not covered.
    tset_lookup = {tset: num for num, tset in enumerate(combinations(range(v), t))}
    ksets = list(enumerate(combinations(range(v), k)))

    # Since flies will be incidence vectors of t-sets, we want a map of k-sets to t-sets they cover.
    # The stopping point will be when score is the number of tsets.
    kset_dict = {num: [tset_lookup[tset] for tset in combinations(kset, t)] for num, kset in ksets}

    def fitness(vector):
        # Figure out what we have covered.
        coverage = [tsetnum for ksetnum, kincidence in enumerate(vector)
                    for tsetnum in kset_dict[ksetnum] if kincidence == 1]

        # If we have covered any t-set multiple times, the score drops to 0 as it is an infeasible solution.
        # Otherwise, it is the number of covered t-sets.
        size = len(coverage)
        # return len(set(coverage))
        if len(set(coverage)) == size:
            return size
        else:
            return 0

    # Instantiate the problem.
    dimensions = len(ksets)
    block_size = (factorial(v) // factorial(t) // factorial(v - t)) //\
                 (factorial(k) // factorial(t) // factorial(k - t))
    solution_size = len(tset_lookup)
    print("Solution size is {}".format(solution_size))
    problem = DispersiveFlies(dimensions, fitness, block_size, solution_size, flies=100, debug=True)

    stats, finished, solution = problem.run()
    return stats, fitness(solution) == solution_size, solution


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {} t k v".format(sys.argv[0]))
        sys.exit(1)
    t, k, v = map(int, sys.argv[1:])

    stats, finished, solution = steiner(t, k, v)
    ksets = list(combinations(range(v), k))
    readable_solution = [ksets[i] for i, value in enumerate(solution) if value == 1]

    if finished:
        for b in readable_solution:
            print(b)
        sys.exit(0)
    else:
        print("Could not finish. Best solution has size {}:".format(len([i for i in solution if i])))
        for b in readable_solution:
            print(b)
        sys.exit(2)
