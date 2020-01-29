##########################################################################

# This is a simple protoss agent using reinforcement learning
# Currently both the agent and the enemy could only do the following actions: send idle probe to harvest mineral, build a pylon, build a gateway,
# train zealots, and attack. 

##########################################################################

from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app
import os
import random
import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, gamma=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # For 90% of the time, agent will pick an action that grants maximum reward; for 10% of the time, agent will pick a random action
    def decide(self, observation, explore=0.1):
        self.check_state_exist(observation)
        if np.random.uniform() > explore:
            state_action = self.q_table.loc[observation].values
            tmp = np.argwhere(state_action == np.max(state_action))
            good_actions = np.reshape(tmp, (tmp.shape[0], ))
            action = self.actions[np.random.choice(good_actions)]
            #state_action = self.q_table.loc[observation:]
            #action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    # Update q_table after each step
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        cur_reward = self.q_table.loc[s, a]
        if s_ != 'terminal':
            fut_reward = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            fut_reward = r
            '''f = open('output.txt', 'a')
            f.write(self.q_table.to_string())
            f.close()'''
        self.q_table.loc[s, a] += self.learning_rate * (fut_reward - cur_reward)

    # Add a state row to the q_table if it does not exist
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class Agent(base_agent.BaseAgent):
    actions = ('do_nothing', 'mine_minerals', 'build_pylon', 'build_gateway', 'train_zealot', 'attack')

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

    # Build a pylon if there is none
    def build_pylon(self, obs):
        pylons = self.get_my_units(obs, units.Protoss.Pylon)
        probes = self.get_my_units(obs, units.Protoss.Probe)
        if len(pylons) == 0 and obs.observation.player.minerals >= 100 and len(probes) >= 1:
            pylon_xy = (22, 26) if self.base_top_left else (35, 42)
            distances = self.get_distances(obs, probes, pylon_xy)
            probe = probes[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Pylon_pt('now', probe.tag, pylon_xy)
        return actions.RAW_FUNCTIONS.no_op()

    # Build a gateway if there is none
    def build_gateway(self, obs):
        gateways = self.get_my_units(obs, units.Protoss.Gateway)
        probes = self.get_my_units(obs, units.Protoss.Probe)
        if len(gateways) == 0 and obs.observation.player.minerals >= 150 and len(probes) >= 1:
            gateway_xy = (22, 21) if self.base_top_left else (35, 45)
            distances = self.get_distances(obs, probes, gateway_xy)
            probe = probes[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Gateway_pt('now', probe.tag, gateway_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_zealot(self, obs):
        comp_gateways = self.get_my_comp_units(obs, units.Protoss.Gateway)
        free_supply = obs.observation.player.food_cap - obs.observation.player.food_used
        if len(comp_gateways) > 0 and obs.observation.player.minerals >= 150 and free_supply >= 2:
            gateway = comp_gateways[0]
            if gateway.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Zealot_quick('now', gateway.tag)
        return actions.RAW_FUNCTIONS.no_op()

    # Order a single zealot to attack the enemy
    def attack(self, obs):
        zealots = self.get_my_units(obs, units.Protoss.Zealot)
        if len(zealots) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            distances = self.get_distances(obs, zealots, attack_xy)
            zealot = zealots[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt('now', zealot.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

class ProtossAgent(Agent):
    def __init__(self):
        super(ProtossAgent, self).__init__()
        self.q_table = QLearningTable(self.actions)
        self.new_game()

    def reset(self):
        super(ProtossAgent, self).reset()
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
        zealots = self.get_my_units(obs, units.Protoss.Zealot)
        queued_zealots = (comp_gateways[0].order_length if len(comp_gateways) > 0 else 0)
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
        ene_zealots = self.get_ene_units(obs, units.Protoss.Zealot)

        return (len(nexuses), len(probes), len(idle_probes), len(pylons), len(comp_pylons), len(gateways), len(comp_gateways), 
                len(zealots), queued_zealots, free_supply, can_build_zealot, can_build_gateway, can_build_pylon, len(ene_probes),
                len(ene_nexuses), len(ene_idle_probes), len(ene_pylons), len(ene_comp_pylons), len(ene_gateways), len(ene_comp_gateways),
                len(ene_zealots))

    def step(self, obs):
        super(ProtossAgent, self).step(obs)
        state = str(self.get_state(obs))
        action = self.q_table.decide(state)
        if self.previous_action is not None:
            self.q_table.learn(self.previous_state, self.previous_action, obs.reward, 'terminal' if obs.last() else state)
        self.previous_state = state
        self.previous_action = action
        #print('test: ' + str(action))
        #print(self.q_table.q_table.to_string())
        return getattr(self, action, 'do_nothing')(obs)

# An agent that randomly chooses an action that we specified for each step
class Enemy(Agent):
    def step(self, obs):
        super(Enemy, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action, 'do_nothing')(obs)

def main(unused_argv):
    agent1 = ProtossAgent()
    agent2 = Enemy()
    try:
        with sc2_env.SC2Env(map_name = 'Simple64', players=[sc2_env.Agent(sc2_env.Race.protoss), sc2_env.Agent(sc2_env.Race.protoss)],
                            agent_interface_format=features.AgentInterfaceFormat(action_space=actions.ActionSpace.RAW, use_raw_units=True, raw_resolution=64),
                            step_mul=48, disable_fog=True) as env:
            run_loop.run_loop([agent1, agent2], env, max_episodes=100)
    except KeyboardInterrupt:
        pass
    f = open('output.txt', 'a')
    f.write(agent1.q_table.q_table.to_string())
    f.close()

if __name__ == '__main__':
    app.run(main)
