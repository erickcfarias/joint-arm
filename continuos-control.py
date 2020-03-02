from unityagents import UnityEnvironment

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch

from agent import *

print('Loading Environment...')
env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64")

# # get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Initialize the Agent
print("Raising Agent...")
# agent = DDPG_Agent(
#     state_size=33,
#     action_size=4,
#     seed=7,
#     buffer_size=int(1e6),
#     batch_size=64,
#     gamma=0.99,
#     tau=0.001,
#     lr=1e-4,
#     w_decay=0.001
#     )

agent = AgentA2C(
    env=env,
    brain_name=brain_name,
    state_size=33,
    act_size=4,
    gamma=0.95,
    critic_lr=1e-4,
    actor_lr=1e-4,
    entropy_beta=1e-2
)

# Train the Agent with DDPG
print("Teaching Agent...")

# scores = agent.train(env, brain_name, 5000)

# Train the Agent with A2C
scores = agent.train(1000)


# Watch a smart agent

# load the weights from file
agent.actor_local.load_state_dict(torch.load('checkpoint.pth'))

env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations[0]

for j in range(200):
    action = agent.act(state)
    # send the action to the environment
    env_info = env.step(action)[brain_name]
    # get the next state
    state = env_info.vector_observations[0]
    # get the reward
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    if done:
        break

env.close()
