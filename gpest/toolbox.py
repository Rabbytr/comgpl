import numpy as np
from deap import gp, tools, algorithms, base
import operator
from .individual import *
from .parameter import Redux
from functools import partial


def std_toolbox(pset, minimize=True):

    Individual = IndividualMin if minimize else IndividualMax

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=Redux.MAX_TREE_HEIGHT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=Redux.MAX_TREE_HEIGHT))
    return toolbox


def cxOnePointListOfTrees(ind1, ind2):
    for tree1, tree2 in zip(ind1, ind2):
        gp.cxOnePoint(tree1, tree2)
    return ind1, ind2


def multitree_toolbox(pset, tree_num, minimize=True):
    def initIndividual(container, func, size):
        return container(gp.PrimitiveTree(func()) for _ in range(size))

    def mutUniformListOfTrees(individual, expr, pset):
        for tree in individual:
            gp.mutUniform(tree, expr=expr, pset=pset)
        return individual,

    Individual = MultiTreeIndividualMin if minimize else MultiTreeIndividualMax

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", initIndividual, Individual, toolbox.expr, size=tree_num)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cxOnePointListOfTrees)
    toolbox.register("mutate", mutUniformListOfTrees, expr=toolbox.expr, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=lambda inds: max([ind.height for ind in inds]),
                                            max_value=Redux.MAX_TREE_HEIGHT))
    toolbox.decorate("mutate", gp.staticLimit(key=lambda inds: max([ind.height for ind in inds]),
                                              max_value=Redux.MAX_TREE_HEIGHT))
    return toolbox


def multipset_toolbox(psets, minimize=True):
    def initIndividual(container, expr, psets):
        return container(gp.PrimitiveTree(expr(pset=pset)) for pset in psets)

    def mutUniformListOfTrees(individual, expr, psets):
        for tree, pset in zip(individual, psets):
            placed_expr = partial(expr, pset=pset)
            gp.mutUniform(tree, expr=placed_expr, pset=pset)
        return individual,

    Individual = MultiTreeIndividualMin if minimize else MultiTreeIndividualMax

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, min_=1, max_=3)
    toolbox.register("individual", initIndividual, MultiTreeIndividualMin, toolbox.expr, psets=psets)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cxOnePointListOfTrees)
    toolbox.register("mutate", mutUniformListOfTrees, expr=toolbox.expr, psets=psets)

    toolbox.decorate("mate", gp.staticLimit(key=lambda inds: max([ind.height for ind in inds]),
                                            max_value=Redux.MAX_TREE_HEIGHT))
    toolbox.decorate("mutate", gp.staticLimit(key=lambda inds: max([ind.height for ind in inds]),
                                              max_value=Redux.MAX_TREE_HEIGHT))
    return toolbox
