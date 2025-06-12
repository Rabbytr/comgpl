from deap import base
from deap import gp

fitnessMin = type('', (base.Fitness,), {'weights': (-1.0,)})
fitnessMax = type('', (base.Fitness,), {'weights': (1.0,)})


class IndividualMin(gp.PrimitiveTree):
    def __init__(self, gene_gen):
        super().__init__(gene_gen)
        self.fitness = fitnessMin()


class IndividualMax(gp.PrimitiveTree):
    def __init__(self, gene_gen):
        super().__init__(gene_gen)
        self.fitness = fitnessMax()


class MultiTreeIndividual(list):
    SPLIT_SYM = '\n'

    def __str__(self):
        return self.SPLIT_SYM.join([str(ind) for ind in self])

    @classmethod
    def from_string(cls, string, pset):
        string_split = string.split(cls.SPLIT_SYM)
        inds = []
        for s in string_split:
            ind = gp.PrimitiveTree.from_string(s, pset)
            inds.append(ind)
        return cls(inds)


class MultiTreeIndividualMax(MultiTreeIndividual):
    def __init__(self, gene_gen):
        super().__init__(gene_gen)
        self.fitness = fitnessMax()

class MultiTreeIndividualMin(MultiTreeIndividual):
    def __init__(self, gene_gen):
        super().__init__(gene_gen)
        self.fitness = fitnessMin()
