import numpy as np
from world.world_cityflow import World
from environment import TSCEnv
import os
from common.registry import Registry


class Evaluation(object):
    def __init__(self, network_name, agent, thread_num=1):

        file_name = f'{network_name}.cfg'
        path = os.path.join('configs/sim/', file_name)

        world = World(path, thread_num=thread_num)

        self.agents = [agent(world, i) for i in range(len(world.intersections))]

        self.env = TSCEnv(world, self.agents, None)

    def _run(self):
        last_obs = self.env.reset()
        action_interval = int(Registry.mapping['model_mapping']['setting'].param['t_min'])
        for _ in range(0, 3600, action_interval):
            actions = []
            last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]
            for idx, ag in enumerate(self.agents):
                actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
            actions = np.stack(actions)
            for i in range(action_interval):
                obs, rewards, dones, _ = self.env.step(actions.flatten())

        return self.env.eng.get_average_travel_time()

    def evaluate(self, func):
        for ag in self.agents:
            ag.rule = func
            ag.reset()
        return self._run()

    def commu_evaluate(self, func, commu_func):
        for ag in self.agents:
            ag.main_rule = func
            ag.commu_rule = commu_func
            ag.reset()
        return self._run()
