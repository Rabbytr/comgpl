import operator
from functools import partial

import numpy as np
from deap import gp


def div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2


def std_pset(input_dim):
    pset = gp.PrimitiveSet("MAIN", arity=input_dim, prefix='x')
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(div, 2)
    # pset.addPrimitive(operator.neg, 1)

    pset.addEphemeralConstant('c', partial(np.random.uniform, -1, 1))
    return pset
