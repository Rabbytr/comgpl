import os
import pickle
import json
import time
from deap import algorithms, tools
from .Utils import seeded_random_state
from .parameter import Redux
from .statistics import std_stats


class Evolution(object):
    def __init__(self, population, ngen, toolbox, pset, seed: int = 42, save_path=None, stats=None):
        if not hasattr(toolbox, 'evaluate'):
            raise AttributeError("Toolbox should have attribute 'evaluate'")

        self.save_path = save_path
        self.seed = seed
        self.save_path = os.path.join(save_path, f'log_{self.seed}') if save_path else None

        self.population = population
        self.toolbox = toolbox
        self.pset = pset
        self.n_gen = ngen
        self.n_pop = len(population)

        self.hof = tools.HallOfFame(Redux.HOF_NUM)

        self.stats = std_stats() if stats is None else stats
        self.logbook = tools.Logbook()
        self.logbook.header = ['time', 'gen', 'nevals'] + self.stats.fields

    def run(self):
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            # os.makedirs(self.history_path, exist_ok=True)
        with seeded_random_state(seed=self.seed):
            for gen in range(self.n_gen):
                self.step(gen)

        self._save_final_state()

    # @property
    # def history_path(self):
    #     return os.path.join(self.save_path, 'history') if self.save_path else None

    def _save_final_state(self):
        if self.save_path is None:
            return
        with open(os.path.join(self.save_path, 'log.pkl'), 'wb') as f:
            pickle.dump(self.logbook, f)
        with open(os.path.join(self.save_path, 'pset.pkl'), 'wb') as f:
            pickle.dump(self.pset, f)
        with open(os.path.join(self.save_path, 'best.txt'), 'w') as f:
            json.dump({
                'expr': str(self.hof[0]),
                'fitness': self.hof[0].fitness.values
            }, f, indent=2)

    def step(self, gen: int):
        fitnesses, invalid_ind = self._eval()

        self._log(gen, len(invalid_ind))
        if gen == self.n_gen - 1: return

        elites = tools.selBest(self.population, k=Redux.N_ELITES)
        offspring = self.toolbox.select(self.population, self.n_pop - Redux.N_ELITES)
        offspring = algorithms.varAnd(offspring, self.toolbox, Redux.CX_PB, Redux.MUT_PB)

        self.population = elites + offspring

    def _eval(self):
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return fitnesses, invalid_ind

    def _log(self, gen: int, n_eval: int):
        self.hof.update(self.population)
        record = self.stats.compile(self.population)
        self.logbook.record(time=time.strftime("%D %H:%M:%S"), gen=gen, nevals=n_eval, **record)
        print(self.logbook.stream, flush=True)
        if self.save_path: self._dump_best_history()

    def _dump_best_history(self):
        with open(os.path.join(self.save_path, 'best_history.txt'), 'a') as f:
            f.write(str(self.hof[0]))
            f.write('\n\n')
