from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app
import os
import random
import numpy as np
import pandas as pd
import cv2

class QTable:
    def __init__(self, actions, path):
        self.actions = actions
        self.q_table = pd.read_csv(path, header=0, index_col=0)

    # For 90% of the time, agent will pick an action that grants maximum reward; for 10% of the time, agent will pick a random action
    def decide(self, observation):
        self.check_state_exist(observation)
        state_action = self.q_table.loc[observation].values
        tmp = np.argwhere(state_action == np.max(state_action))
        good_actions = np.reshape(tmp, (tmp.shape[0], ))
        action = self.actions[np.random.choice(good_actions)]
        return action

    # Add a state row to the q_table if it does not exist
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class Agent(base_agent.BaseAgent):
    actions = ('do_nothing', 'mine_minerals', 'mine_gas', 'build_assimilator', 'build_pylon', 'build_gateway', 'build_cybercore', 'train_probe', 'train_zealot', 'train_stalker', 'harass', 'attack_all')

    def step(self, obs):
        super(Agent, self).step(obs)
        if obs.first(): # Determine the starting location
            nexus = self.get_my_units(obs, units.Protoss.Nexus)[0]
            self.base_top_left = (nexus.x < 32)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def get_my_units(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type 
        and unit.alliance == features.PlayerRelative.SELF]

    def get_my_comp_units(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type and unit.build_progress == 100 
        and unit.alliance == features.PlayerRelative.SELF]

    def get_ene_units(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type and unit.alliance == features.PlayerRelative.ENEMY]

    def get_ene_comp_units(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type and unit.build_progress == 100 
        and unit.alliance == features.PlayerRelative.ENEMY]

    # Get distances between units and a specified coordinate
    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    # Get an idle probe to harvest minerals
    def mine_minerals(self, obs):
        probes = self.get_my_units(obs, units.Protoss.Probe)
        idle_probes = [probe for probe in probes if probe.order_length==0]
        if len(idle_probes) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units 
                                if unit.unit_type in [units.Neutral.BattleStationMineralField,
                                                       units.Neutral.BattleStationMineralField750,
                                                       units.Neutral.LabMineralField,
                                                       units.Neutral.LabMineralField750,
                                                       units.Neutral.MineralField,
                                                       units.Neutral.MineralField750,
                                                       units.Neutral.PurifierMineralField,
                                                       units.Neutral.PurifierMineralField750,
                                                       units.Neutral.PurifierRichMineralField,
                                                       units.Neutral.PurifierRichMineralField750,
                                                       units.Neutral.RichMineralField,
                                                       units.Neutral.RichMineralField750]]
            probe = random.choice(idle_probes)
            distances = self.get_distances(obs, mineral_patches, (probe.x, probe.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit('now', probe.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    # Order a probe to mine gas if there exists a assimilator where less than 3 probes are assigned to it
    def mine_gas(self, obs):
        probes = self.get_my_units(obs, units.Protoss.Probe)
        assimilators = self.get_my_units(obs, units.Protoss.Assimilator)
        assimilators = [assimilator for assimilator in assimilators if assimilator.assigned_harvesters < 3]
        if len(probes) > 0 and len(assimilators) > 0:
            probe = random.choice(probes)
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit('now', probe.tag, assimilators[0].tag)
        return actions.RAW_FUNCTIONS.no_op()

    # Build an assimilator if total number of assimilators is less than 2
    def build_assimilator(self, obs):
        assimilators = self.get_my_units(obs, units.Protoss.Assimilator)
        probes = self.get_my_units(obs, units.Protoss.Probe)
        if len(assimilators) < 2 and obs.observation.player.minerals >= 75 and len(probes) >= 1:
            assimilator_xy = [unit for unit in obs.observation.raw_units 
                                if unit.unit_type in [units.Neutral.PurifierVespeneGeyser,
                                                       units.Neutral.RichVespeneGeyser,
                                                       units.Neutral.ShakurasVespeneGeyser,
                                                       units.Neutral.VespeneGeyser]]
            probe = random.choice(probes)
            distances = self.get_distances(obs, assimilator_xy, (probe.x, probe.y))
            assimilator = assimilator_xy[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Assimilator_unit('now', probe.tag, assimilator.tag)
        return actions.RAW_FUNCTIONS.no_op()

    # Build a pylon if there are less than 4 pylons
    def build_pylon(self, obs):
        pylons = self.get_my_units(obs, units.Protoss.Pylon)
        probes = self.get_my_units(obs, units.Protoss.Probe)
        if len(pylons) < 4 and obs.observation.player.minerals >= 100 and len(probes) >= 1:
            if len(pylons) == 0:
                pylon_xy = (22, 20) if self.base_top_left else (36, 41)
            elif len(pylons) == 1:
                pylon_xy = (22, 22) if self.base_top_left else (36, 43)
            elif len(pylons) == 2:
                pylon_xy = (22, 24) if self.base_top_left else (36, 45)
            else:
                pylon_xy = (22, 26) if self.base_top_left else (36, 47)
            distances = self.get_distances(obs, probes, pylon_xy)
            probe = probes[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Pylon_pt('now', probe.tag, pylon_xy)
        return actions.RAW_FUNCTIONS.no_op()

    # Build a gateway if there are less than 2 gateways
    def build_gateway(self, obs):
        gateways = self.get_my_units(obs, units.Protoss.Gateway)
        probes = self.get_my_units(obs, units.Protoss.Probe)
        if len(gateways) < 2 and obs.observation.player.minerals >= 150 and len(probes) >= 1:
            if len(gateways) == 0:
                gateway_xy = (24, 26) if self.base_top_left else (34, 41)
            else:
                gateway_xy = (24, 22) if self.base_top_left else (34, 46)
            distances = self.get_distances(obs, probes, gateway_xy)
            probe = probes[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Gateway_pt('now', probe.tag, gateway_xy)
        return actions.RAW_FUNCTIONS.no_op()

    # Build a cybernetics core if there is none
    def build_cybercore(self, obs):
        cybercore = self.get_my_units(obs, units.Protoss.CyberneticsCore)
        probes = self.get_my_units(obs, units.Protoss.Probe)
        if len(cybercore) == 0 and obs.observation.player.minerals >= 150 and len(probes) >= 1:
            cybercore_xy = (24, 24) if self.base_top_left else(34, 44)
            distances = self.get_distances(obs, probes, cybercore_xy)
            probe = probes[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_CyberneticsCore_pt('now', probe.tag, cybercore_xy)
        return actions.RAW_FUNCTIONS.no_op()

    # Train a probe if a nexus does not have the optimal number of probes (16)
    def train_probe(self, obs):
        nexuses = self.get_my_units(obs, units.Protoss.Nexus)
        nexuses = [nexus for nexus in nexuses if nexus.assigned_harvesters < 16]
        if len(nexuses) > 0 and obs.observation.player.minerals >= 50:
            nexus = nexuses[0]
            if nexus.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Probe_quick('now', nexus.tag)
        return actions.RAW_FUNCTIONS.no_op()

    # Pick a gateway that has the least queued training
    def pick_gateway(self, obs, gateways):
        gateway = gateways[0]
        for i in gateways:
            if i.order_length < gateway.order_length:
                gateway = i
        return gateway

    def train_zealot(self, obs):
        free_supply = obs.observation.player.food_cap - obs.observation.player.food_used
        if obs.observation.player.minerals < 150 or free_supply < 2:
            return actions.RAW_FUNCTIONS.no_op()
        comp_gateways = self.get_my_comp_units(obs, units.Protoss.Gateway)
        comp_warpgate = self.get_my_comp_units(obs, units.Protoss.WarpGate)
        if len(comp_warpgate) > 0:
            for i in comp_warpgate:
                if i.order_progress_0 == 0:
                    if self.base_top_left:
                        loc = [i.x+2, i.y]
                    else:
                        loc = [i.x-2, i.y]
                    return actions.RAW_FUNCTIONS.TrainWarp_Zealot_pt('now', i.tag, loc)
        if len(comp_gateways) > 0:
            gateway = self.pick_gateway(obs, comp_gateways) 
            if gateway.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Zealot_quick('now', gateway.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_stalker(self, obs):
        comp_gateways = self.get_my_comp_units(obs, units.Protoss.Gateway)
        comp_warpgate = self.get_my_comp_units(obs, units.Protoss.WarpGate)
        comp_cybercore = self.get_my_comp_units(obs, units.Protoss.CyberneticsCore)
        free_supply = obs.observation.player.food_cap - obs.observation.player.food_used
        if len(comp_cybercore) < 1 or obs.observation.player.minerals < 125 or obs.observation.player.vespene < 50 or free_supply < 2:
            return actions.RAW_FUNCTIONS.no_op()
        if len(comp_warpgate) > 0:
            for i in comp_warpgate:
                if i.order_progress_0 == 0:
                    if self.base_top_left:
                        loc = [i.x+2, i.y]
                    else:
                        loc = [i.x-2, i.y]
                    return actions.RAW_FUNCTIONS.TrainWarp_Stalker_pt('now', i.tag, loc)
        if len(comp_gateways) > 0:
            gateway = self.pick_gateway(obs, comp_gateways)
            if gateway.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Stalker_quick('now', gateway.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def get_inbase_army(self, obs):
        zealots = self.get_my_units(obs, units.Protoss.Zealot)
        stalkers = self.get_my_units(obs, units.Protoss.Stalker)
        tags = []
        if len(zealots) > 0:
            for i in zealots:
                if self.base_top_left and i.x >=24 and i.x <= 26 and i.y >= 22 and i.y <= 27 and i.order_length == 0:
                    tags.append(i.tag)
                if not self.base_top_left and i.x >= 32 and i.x <= 34 and i.y >= 41 and i.y <= 46 and i.order_length == 0:
                    tags.append(i.tag)
        if len(stalkers) > 0:
            for i in stalkers:
                if self.base_top_left and i.x >=24 and i.x <= 26 and i.y >= 22 and i.y <= 26 and i.order_length == 0:
                    tags.append(i.tag)
                if not self.base_top_left and i.x >= 32 and i.x <= 34 and i.y >= 41 and i.y <= 46 and i.order_length == 0:
                    tags.append(i.tag)
        return tags

    def get_entire_army(self, obs):
        zealots = self.get_my_units(obs, units.Protoss.Zealot)
        stalkers = self.get_my_units(obs, units.Protoss.Stalker)
        tags = []
        if len(zealots) > 0:
            for i in zealots:
                tags.append(i.tag)
        if len(stalkers) > 0:
            for i in stalkers:
                tags.append(i.tag)
        return tags

    def move_inbase_army(self, obs, tags):
        if self.base_top_left:
            return actions.RAW_FUNCTIONS.Move_pt('now', tags, [27, 23])
        else:
            return actions.RAW_FUNCTIONS.Move_pt('now', tags, [30, 45])

    # Attack enemy with 3 units when there are at least 3 attack units. Attack coordinate set to be the nearest enemy nexus
    def harass(self, obs):
        army = self.get_entire_army(obs)
        if len(army) >= 3:
            nexuses = self.get_my_units(obs, units.Protoss.Nexus)
            ene_nexuses = self.get_ene_units(obs, units.Protoss.Nexus)
            if len(nexuses) > 0 and len(ene_nexuses) > 0:
                distances = self.get_distances(obs, ene_nexuses, (nexuses[0].x, nexuses[0].y))
                attack_point = ene_nexuses[np.argmin(distances)]
                attack_xy = (attack_point.x, attack_point.y)
            else:
                attack_xy = (38, 44) if self.base_top_left else (19, 23)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt('now', np.random.choice(army, size=3, replace=False), (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    # Attack the enemy with the whole army when there are at least 6 units. Attack coordinate set to be the nearest enemy nexus
    def attack_all(self, obs):
        army = self.get_entire_army(obs)
        if len(army) >= 6:
            nexuses = self.get_my_units(obs, units.Protoss.Nexus)
            ene_nexuses = self.get_ene_units(obs, units.Protoss.Nexus)
            if len(nexuses) > 0 and len(ene_nexuses) > 0:
                distances = self.get_distances(obs, ene_nexuses, (nexuses[0].x, nexuses[0].y))
                attack_point = ene_nexuses[np.argmin(distances)]
                attack_xy = (attack_point.x, attack_point.y)
            else:
                attack_xy = (38, 44) if self.base_top_left else (19, 23)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt('now', army, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

class ProtossAgent(Agent):
    def __init__(self):
        super(ProtossAgent, self).__init__()
        self.q_table = QTable(self.actions, 'q-table2.csv')
        self.image_data = []
        self.image_index = 0
        self.action_hist = []
        self.warp_gate = False
        self.warp_gate_complete = False
        self.new_game()

    def reset(self):
        super(ProtossAgent, self).reset()
        self.warp_gate = False
        self.warp_gate_complete = False
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

    def get_state(self, obs):
        probes = self.get_my_units(obs, units.Protoss.Probe)
        idle_probes = [probe for probe in probes if probe.order_length == 0]
        nexuses = self.get_my_units(obs, units.Protoss.Nexus)
        pylons = self.get_my_units(obs, units.Protoss.Pylon)
        comp_pylons = self.get_my_comp_units(obs, units.Protoss.Pylon)
        gateways = self.get_my_units(obs, units.Protoss.Gateway)
        comp_gateways = self.get_my_comp_units(obs, units.Protoss.Gateway)
        cybercore = self.get_my_units(obs, units.Protoss.CyberneticsCore)
        comp_cybercore = self.get_my_comp_units(obs, units.Protoss.CyberneticsCore)
        zealots = self.get_my_units(obs, units.Protoss.Zealot)
        stalkers = self.get_my_units(obs, units.Protoss.Stalker)
        #queued_zealots = (comp_gateways[0].order_length if len(comp_gateways) > 0 else 0)
        free_supply = obs.observation.player.food_cap - obs.observation.player.food_used

        can_build_pylon = obs.observation.player.minerals >= 100
        can_build_gateway = obs.observation.player.minerals >= 150
        can_build_zealot = obs.observation.player.minerals >= 100

        ene_probes = self.get_ene_units(obs, units.Protoss.Probe)
        ene_idle_probes = [probe for probe in ene_probes if probe.order_length == 0]
        ene_nexuses = self.get_ene_units(obs, units.Protoss.Nexus)
        ene_pylons = self.get_ene_units(obs, units.Protoss.Pylon)
        ene_comp_pylons = self.get_ene_comp_units(obs, units.Protoss.Pylon)
        ene_gateways = self.get_ene_units(obs, units.Protoss.Gateway)
        ene_comp_gateways = self.get_ene_comp_units(obs, units.Protoss.Gateway)
        ene_cybercore = self.get_ene_units(obs, units.Protoss.CyberneticsCore)
        ene_comp_cybercore = self.get_ene_comp_units(obs, units.Protoss.CyberneticsCore)
        ene_zealots = self.get_ene_units(obs, units.Protoss.Zealot)
        ene_stalkers = self.get_ene_units(obs, units.Protoss.Stalker)

        '''
        return (len(nexuses), len(probes), len(idle_probes), len(pylons), len(comp_pylons), len(gateways), len(comp_gateways), len(cybercore),
                len(comp_cybercore), len(zealots), len(stalkers), queued_zealots, free_supply, can_build_zealot, can_build_gateway, can_build_pylon, len(ene_probes),
                len(ene_nexuses), len(ene_idle_probes), len(ene_pylons), len(ene_comp_pylons), len(ene_gateways), len(ene_comp_gateways),
                len(ene_cybercore), len(ene_comp_cybercore), len(ene_zealots), len(ene_stalkers))
        '''

        return (len(probes), len(comp_gateways), 
                len(comp_cybercore), (len(zealots) + len(stalkers)), free_supply, 
                len(ene_comp_gateways),
                len(ene_comp_cybercore), (len(ene_zealots) + len(ene_stalkers)))

    def get_image(self, obs):
        game_map = np.full((64, 64, 3), 255, np.uint8)
        #game_map = np.zeros((64, 64, 3), np.uint8)
        nexuses = self.get_my_units(obs, units.Protoss.Nexus)
        comp_pylons = self.get_my_comp_units(obs, units.Protoss.Pylon)
        comp_gateways = self.get_my_comp_units(obs, units.Protoss.Gateway)
        comp_warpgate = self.get_my_comp_units(obs, units.Protoss.WarpGate)
        comp_cybercore = self.get_my_comp_units(obs, units.Protoss.CyberneticsCore)
        zealots = self.get_my_units(obs, units.Protoss.Zealot)
        stalkers = self.get_my_units(obs, units.Protoss.Stalker)
        ene_nexuses = self.get_ene_units(obs, units.Protoss.Nexus)
        ene_comp_pylons = self.get_ene_comp_units(obs, units.Protoss.Pylon)
        ene_comp_gateways = self.get_ene_comp_units(obs, units.Protoss.Gateway)
        ene_comp_warpgate = self.get_ene_comp_units(obs, units.Protoss.WarpGate)
        ene_comp_cybercore = self.get_ene_comp_units(obs, units.Protoss.CyberneticsCore)
        ene_zealots = self.get_ene_units(obs, units.Protoss.Zealot)
        ene_stalkers = self.get_ene_units(obs, units.Protoss.Stalker)
        for i in nexuses:
            cv2.circle(game_map, (int(i.x), int(i.y)), 3, (0, 255, 0), -1) #Green
        for i in comp_pylons:
            cv2.circle(game_map, (int(i.x), int(i.y)), 1, (0, 255, 255), -1) 
        for i in comp_gateways:
            cv2.circle(game_map, (int(i.x), int(i.y)), 2, (255, 128, 0), -1)
        for i in comp_warpgate:
            cv2.circle(game_map, (int(i.x), int(i.y)), 2, (255, 128, 0), -1) # Same color as gateway
        for i in comp_cybercore:
            cv2.circle(game_map, (int(i.x), int(i.y)), 1, (0, 102, 204), -1)
        for i in zealots:
            cv2.circle(game_map, (int(i.x), int(i.y)), 1, (255, 255, 0), -1) #Yellow
        for i in stalkers:
            cv2.circle(game_map, (int(i.x), int(i.y)), 1, (0, 128, 255), -1)
        for i in ene_nexuses:
            cv2.circle(game_map, (int(i.x), int(i.y)), 3, (255, 0, 0), -1) #Red
        for i in ene_comp_pylons:
            cv2.circle(game_map, (int(i.x), int(i.y)), 1, (255, 102, 102), -1)
        for i in ene_comp_gateways:
            cv2.circle(game_map, (int(i.x), int(i.y)), 2, (153, 0, 76), -1)
        for i in ene_comp_warpgate:
            cv2.circle(game_map, (int(i.x), int(i.y)), 2, (153, 0, 76), -1)
        for i in ene_comp_cybercore:
            cv2.circle(game_map, (int(i.x), int(i.y)), 1, (102, 0, 51), -1)
        for i in ene_zealots:
            cv2.circle(game_map, (int(i.x), int(i.y)), 1, (127, 0, 255), -1) #Light Purple
        for i in ene_stalkers:
            cv2.circle(game_map, (int(i.x), int(i.y)), 1, (51, 0, 102), -1) #Dark Purple
        #cv2.imshow('test', game_map)
        #cv2.waitKey(1)
        #cv2.destroyAllWindows()
        return game_map

    def step(self, obs):
        super(ProtossAgent, self).step(obs)
        probes = self.get_my_units(obs, units.Protoss.Probe)
        idle_probes = [probe for probe in probes if probe.order_length == 0]
        inbase_army_tags = self.get_inbase_army(obs)
        comp_gateways = self.get_my_comp_units(obs, units.Protoss.Gateway)
        comp_cybercore = self.get_my_comp_units(obs, units.Protoss.CyberneticsCore)
        if len(comp_cybercore) >= 1 and self.warp_gate: # Check if warp_gate research is complete
            if comp_cybercore[0].order_progress_0 == 0:
                self.warp_gate_complete = True
        if len(inbase_army_tags) > 0:
            return self.move_inbase_army(obs, inbase_army_tags)
        if len(idle_probes) > 0: # If there is any idle probe, send them to mine minerals for this step
            return getattr(self, 'mine_minerals', 'do_nothing')(obs)
        # Research warp gate if it hasn't been researched and we have a cyber core
        if len(comp_cybercore) >= 1 and not self.warp_gate and obs.observation.player.minerals >= 50 and obs.observation.player.vespene >= 50:
            tag = comp_cybercore[0].tag
            self.warp_gate = True
            return actions.RAW_FUNCTIONS.Research_WarpGate_quick('now', tag)
        if self.warp_gate_complete and len(comp_gateways) >= 1: # If warp_gate research is complete, morph all gateways
            tags = []
            for i in comp_gateways:
                tags.append(i.tag)
            return actions.RAW_FUNCTIONS.Morph_WarpGate_quick('now', tags)

        state = str(self.get_state(obs))
        action = self.q_table.decide(state)
        image = self.get_image(obs)
        self.image_data.append(image)
        self.action_hist.append(action)
        if obs.last():
            if obs.reward == 1:
                for i in range(len(self.image_data)):
                    cv2.imwrite('C:/Users/Yucheng/PycharmProjects/TF2/sc2/train/train image/' + str(self.image_index) + '.png', self.image_data[i])
                    self.image_index += 1
                f = open('C:/Users/Yucheng/PycharmProjects/TF2/sc2/train/actions.csv', 'a')
                np.savetxt(f, np.array(self.action_hist), fmt='%s')
                f.close()
            self.image_data = [] #Reset image for new game
            self.action_hist = [] #Reset action history for new game
        return getattr(self, action, 'do_nothing')(obs)

# An agent that randomly chooses an action that we specified for each step
class Enemy(Agent):
    def step(self, obs):
        super(Enemy, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action, 'do_nothing')(obs)

def main(unused_argv):
    agent1 = ProtossAgent()
    mode = input('Set opponent as default AI? Type Yes or No: ')
    if mode == 'No':
        agent2 = Enemy()
        try:
            with sc2_env.SC2Env(map_name = 'Simple64', players=[sc2_env.Agent(sc2_env.Race.protoss), sc2_env.Agent(sc2_env.Race.protoss)],
                                agent_interface_format=features.AgentInterfaceFormat(action_space=actions.ActionSpace.RAW, use_raw_units=True, raw_resolution=64),
                                step_mul=40, disable_fog=True, realtime=False) as env:
                run_loop.run_loop([agent1, agent2], env, max_episodes=3)
        except KeyboardInterrupt:
            pass
    else:
        try:
            with sc2_env.SC2Env(map_name = 'Simple64', players=[sc2_env.Agent(sc2_env.Race.protoss), sc2_env.Bot(sc2_env.Race.protoss, sc2_env.Difficulty.easy)],
                                agent_interface_format=features.AgentInterfaceFormat(action_space=actions.ActionSpace.RAW, use_raw_units=True, raw_resolution=64),
                                step_mul=40, disable_fog=True, realtime=False) as env:
                run_loop.run_loop([agent1], env, max_episodes=150)
        except KeyboardInterrupt:
            pass 

if __name__ == '__main__':
    app.run(main)





#actions = ('do_nothing', 'mine_minerals', 'mine_gas', 'build_assimilator', 'build_pylon', 'build_gateway', 'build_cybercore', 'train_probe', 'train_zealot', 'train_stalker', 'harass', 'attack_all')
#test = QTable(actions, 'testcsv.csv')

