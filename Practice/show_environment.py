import time
from typing import Type

import torch

from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video
from my_transport import MyTransportScenario


def show_env():
    
    env_kwargs = {}
    n_steps = 1000
    n_envs = 2
    
    
    policy = RandomPolicy(continuous_action=True)
    

    env = make_env(
        scenario=MyTransportScenario(),
        num_envs=n_envs,
        device="cpu",
        continuous_actions=True,
        wrapper=None,
        **env_kwargs,
    )
    
    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    obs = env.reset()
    for _ in range(n_steps):
        step += 1
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
        obs, rews, dones, info = env.step(actions)
        print(rews)
        frame_list.append(
            env.render(
                mode="rgb_array",
                agent_index_focus=None,
                visualize_when_rgb=True,
            )
        )
    
    # save_video("transport", frame_list, 1 / env.scenario.world.dt)
        
if __name__ == "__main__":
    show_env()