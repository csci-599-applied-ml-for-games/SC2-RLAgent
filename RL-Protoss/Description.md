# Training Protoss bot with q-learning
## Both the agent and the enemy are restricted to the same list of pre-defined actions

- Version 1: Pre-defined actions: send idle probe to mine minerals, build a pylon, build a gateway, train zealots, attack with a single zealot
- Version 2-0: Pre-defined actions: train probes, send idle probe to mine minerals, send probe to harvest gas, build assimilators, build up to 4 pylons, build up to 2 gateways, build a cybernectics core, train zealots, train stalkers, attack with exact 3 units, attack with the entire army
- Version 2-1: Same set of actions as 2-0. Changes: When making a decision, agent will prioritize sending idle probe to mine minerals if there exists any idle drone; attack coordinate set to be opponent's nearest nexus instead of a pre-defined coordinate. Therefore, if the opponent's nexus changes location, our army could still attack its location. 


Challenges:
- All the build locations are hard coded coordinates. Is there a way to select building location intelligently? Perhaps we could use image recognition to search for locations that could be used to build structures? 
- Current army control is extremely simple: we directly attack the enemy's base. How could we add more army actions such as formation, chase, kite, surround, defend, etc.
- How do we decide when to engage enemies? Perhaps with evaluation functions? 
- How will we detect cloaked enemies?
- Is there a way to reduce the size of q-learning table? As we add more actions to the agent, the size of the q table will grow significantly.
