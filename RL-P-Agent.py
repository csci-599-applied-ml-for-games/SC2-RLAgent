from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app
import os
import random
import numpy as np

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
			actions = self.q_table.loc[observation:]
			action = np.random.choice(actions[actions==np.max(actions)].index)
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
		self.q_table.loc[s, a] += self.learning_rate * (fut_reward - cur_reward)

	# Add a state row to the q_table if it does not exist
	def check_state_exist(self, state):
		if state not in self.q_table.index:
			self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
