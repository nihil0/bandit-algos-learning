from mab import BernoulliBandit
from agents import IntoxicatedAgent

bandit = BernoulliBandit(3, [0.25, 0.5, 0.75])

agent = IntoxicatedAgent(bandit, horizon=10000)

agent.run()

print(agent.data.groupby("actions").mean())
print(f"Total reward: {agent.data.rewards.sum()}")