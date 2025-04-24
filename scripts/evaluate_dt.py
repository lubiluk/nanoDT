import numpy as np
from nanodt.agent import NanoDTAgent
import gymnasium as gym


def evaluate():
    env = gym.make("HalfCheetah-v4")
    agent = NanoDTAgent.load("output/dt/minari-halfcheetah-medium-v0.pth")

    for _ in range(5):
        agent.reset(target_return=6000)
        obs, info = env.reset()
        done = False
        accumulated_rew = 0
        while not done:
            action = agent.act(obs)
            obs, rew, ter, tru, info = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        print("Accumulated rew: ", accumulated_rew)
        # print("Success: ", info["is_success"])


if __name__ == "__main__":
    evaluate()
