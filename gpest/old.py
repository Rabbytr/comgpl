import json
import os
import pickle
from deap import gp, tools, algorithms, base
import deap
import operator
from .toolbox import std_toolbox
from .statistics import std_stats
from .Utils import seeded_random_state
from .parameter import Redux


class Evolution(object):
    def __init__(self, n_pop, n_gen, pset, seed: int = 42, save_path=None):
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.save_path = save_path
        self.seed = seed

        self.population = None
        self.toolbox = None

    def run(self):
        if self.save_path:
            os.makedirs(self.history_path, exist_ok=True)
        with seeded_random_state(seed=self.seed):
            self.population = self.toolbox.population(n=self.n_pop)
            for gen in range(self.n_gen):
                self.step(gen)

        if self.save_path:
            self._save_final_state()

    @property
    def history_path(self):
        return os.path.join(self.save_path, 'history') if self.save_path else None

    def _save_final_state(self):
        raise NotImplementedError

    def step(self, gen: int):
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        self._log(gen, len(invalid_ind))

        elites = deap.tools.selBest(self.population, k=Redux.N_ELITES)
        # Vary the pool of individuals
        offspring = self.toolbox.select(self.population, self.n_pop - Redux.N_ELITES)
        offspring = algorithms.varAnd(offspring, self.toolbox, Redux.CX_PB, Redux.MUT_PB)

        self.population = elites + offspring

    def _log(self, gen: int, n_eval: int):
        raise NotImplementedError

    def set_eval(self, func):
        self.toolbox.register("evaluate", func, pset=self.pset)

    def register(self, *args, **kwargs):
        self.toolbox.register(*args, **kwargs)


class GpEvolution(Evolution):
    def __init__(self, n_pop, n_gen, pset, seed: int = 42, save_path=None):
        self.n_pop = n_pop
        self.n_gen = n_gen

        if not isinstance(seed, int):
            raise TypeError('Seed needs to be a int')
        self.seed = seed

        self.pset = pset
        self.toolbox = std_toolbox(pset)
        self.population = None
        self.stats = std_stats()
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + self.stats.fields
        # self.logbook.header = self.stats.fields if self.stats else []
        self.hof = tools.HallOfFame(Redux.HOF_NUM)

        self.total_eval_num = 0
        self.save_path = os.path.join(save_path, f'log_{self.seed}') if save_path else None

    def _save_final_state(self):
        with open(os.path.join(self.save_path, 'pset.pkl'), 'wb') as f:
            pickle.dump(self.pset, f)
        with open(os.path.join(self.save_path, 'log.pkl'), 'wb') as f:
            pickle.dump(self.logbook, f)
        with open(os.path.join(self.save_path, 'best.txt'), 'w') as f:
            json.dump({
                'expr': str(self.hof[0]),
                'fitness': self.hof[0].fitness.values
            }, f, indent=2)

    def _log(self, gen: int, n_eval: int):
        if self.hof is not None:
            self.hof.update(self.population)

        record = self.stats.compile(self.population) if self.stats else {}
        self.logbook.record(gen=gen, nevals=n_eval, **record)
        # if verbose:
        print(self.logbook.stream, flush=True)

        self.total_eval_num += n_eval

        self._write_history(f'generation_{gen}.txt', self.population)

    def _write_history(self, file_name, individuals):
        if self.save_path is None: return

        with open(os.path.join(self.history_path, file_name), 'w') as f:
            f.write(f'{self.n_pop}\n')
            for i, ind in enumerate(individuals):
                f.write(f"{i} | {','.join([str(fit) for fit in ind.fitness.values])} | {str(ind)}")
                f.write('\n')
