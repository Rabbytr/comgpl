import json
import os
import pickle
import numpy as np
from deap import gp, base
from .Utils import load_pickle


def to_individual(expr, fitness, pset, compile=False):
    ind = gp.PrimitiveTree.from_string(expr, pset=pset)
    setattr(ind, 'fitness', type('', (base.Fitness,), {'weights': (1,)})())
    ind.fitness.values = fitness
    if compile:
        func = gp.compile(expr, pset)
    else:
        func = None
    return ind, func


class Analyzer(object):
    def __init__(self, path):
        self.path = path
        self.sub_paths = [os.path.join(self.path, sub) for sub in os.listdir(self.path)]

    def converge_data(self):
        data = []
        for subp in self.sub_paths:
            with open(os.path.join(subp, 'log.pkl'), 'rb') as f:
                log = pickle.load(f)
            data.append(log.chapters['fitness'].select('avg'))
        return np.array(data)

    def generations(self, sub_path):
        history_path = os.path.join(sub_path, 'history')

        pset = load_pickle(os.path.join(sub_path, 'pset.pkl'))

        generation_files = [file for file in os.listdir(history_path) if file.endswith(".txt")]

        for i in range(len(generation_files)):
            pop_path = os.path.join(history_path, f'generation_{i}.txt')
            population = self._get_population(pop_path, pset)
            yield population

    def best_individuals(self, compile=False):
        inds = []
        for subp in self.sub_paths:
            pset = load_pickle(os.path.join(subp, 'pset.pkl'))
            with open(os.path.join(subp, 'best.txt'), 'r') as f:
                best_json = json.load(f)
            ind, func = to_individual(**best_json, pset=pset, compile=compile)
            if compile:
                inds.append(func)
            else:
                inds.append(ind)
        return inds

    def _get_population(self, pop_path, pset):
        population = []
        with open(pop_path, 'r') as f:
            n_pop = int(f.readline().strip())
            for _ in range(n_pop):
                line = [i.strip() for i in f.readline().split('|')]
                expr = line[-1]
                fitness = tuple(float(i) for i in line[1:-1])
                ind = to_individual(expr, fitness, pset)
                population.append(ind)
        return population
