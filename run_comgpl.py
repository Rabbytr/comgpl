import agent
from deap import gp
from common.registry import Registry
from pathos.multiprocessing import ProcessPool
import argparse
from gpest.evolution import Evolution
from common import interface
from utils.logger import build_config
from gpest.psets import std_pset
from gpest.toolbox import multitree_toolbox
from Evaluation import Evaluation

# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=1, help='number of threads')  # used in cityflow
parser.add_argument('--seed', type=int, default=1, help="seed algorithem")
parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")
parser.add_argument('--agent', type=str, default='comgpl', help="seed algorithem")
parser.add_argument('-n', '--network', type=str, default='MHT', help="network name")

args = parser.parse_args()

config, _ = build_config(args)
interface.ModelAgent_param_Interface(config)

Agent = Registry.mapping['model_mapping'][args.agent]

model = Evaluation(args.network, agent=Agent, thread_num=args.thread_num)

def evalSymbReg(individual, pset):
    func = gp.compile(individual[0], pset)
    commu_func = gp.compile(individual[1], pset)
    return model.commu_evaluate(func, commu_func),


if __name__ == '__main__':
    pset = std_pset(Agent.ARITY)
    toolbox = multitree_toolbox(pset=pset, tree_num=2, minimize=True)

    toolbox.register('evaluate', evalSymbReg, pset=pset)
    pool = ProcessPool(5)
    toolbox.register('map', pool.map)
    # toolbox.register('map', map)

    from gpest.parameter import Redux
    Redux.HOF_NUM = 3

    from gpest.Utils import seeded_random_state
    with seeded_random_state(args.seed):
        pop = toolbox.population(n=5)
    g = Evolution(population=pop, ngen=5, toolbox=toolbox, pset=pset, seed=args.seed, save_path=None)
    g.run()

    for ind in g.hof[0]:
        print(ind)
