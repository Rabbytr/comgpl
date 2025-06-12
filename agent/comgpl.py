from . import GPLight
from common.registry import Registry
import os
from collections import defaultdict
from CityflowEnvExt import upstream_features

@Registry.register_model('comgpl')
class ComGPL(GPLight):
    ARITY = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_edge_intersection = None

        self.rule = 5
        self.main_rule = None
        self.commu_rule = None

    def reset(self):
        super().reset()
        self.is_edge_intersection = False
        for in_road in self.inter_obj.in_roads:
            nbr_inter = self.world.id2intersection.get(in_road['startIntersection'])
            if nbr_inter is None:
                self.is_edge_intersection = True

        if self.is_edge_intersection:
            self.rule = self.main_rule
        else:
            self.rule = self.commu_rule

    def get_action(self, ob, phase, test=True):
        lvc = self.world.get_info("lane_count")
        lvw = self.world.get_info("lane_waiting_count")
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase

        commu_lvc = upstream_features(self.inter_obj, self.world, lvc)
        commu_lvw = upstream_features(self.inter_obj, self.world, lvw)

        max_urgency = None
        action = -1
        for phase_id in range(len(self.inter_obj.phases)):
            lanelinks = [(start, end)
                 for start, end in self.inter_obj.phase_available_lanelinks[phase_id]
                 if not start.endswith('2')]

            list_dict = defaultdict(list)
            for key, value in lanelinks:
                list_dict[key].append(value)
            TMs = [[key] + sorted(values) for key, values in list_dict.items()]

            urgency = 0.0
            for tm in TMs:
                t = [lvw[lane] for lane in tm]
                t.append(commu_lvw[tm[0]])
                t.extend([lvc[lane] for lane in tm])
                t.append(commu_lvc[tm[0]])

                urgency += self.rule(*t)

            if max_urgency is None or urgency > max_urgency:
                action = phase_id
                max_urgency = urgency

        return action

