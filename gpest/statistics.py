import numpy as np
from deap import tools


def std_stats():
    mstats = tools.Statistics(lambda ind: ind.fitness.values[0])
    # stats_size = tools.Statistics(lambda ind: ind.height)
    # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    # mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats


def m_stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    # mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats
