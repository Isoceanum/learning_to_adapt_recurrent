from learning_to_adapt.envs.normalized_env import normalize
from learning_to_adapt.envs import RoverEnv
import numpy as np

def main():
    # Create normalized Rover environment (same style as working script)
    env = normalize(RoverEnv(reset_every_episode=True, task=None))

    obs = env.reset()
    sim = env.wrapped_env.sim  # Access MuJoCo sim inside wrapped env
    xpos_before = sim.data.qpos[0]  # x position before action

    print("Initial observation:", obs)

    for step in range(10):
        action = env.action_space.sample()  # random action
        obs, reward, done, _ = env.step(action)

        xpos_after = sim.data.qpos[0]
        delta_x = xpos_after - xpos_before
        xpos_before = xpos_after

        print(f"Step {step+1}:")
        print(f"  Action: {np.round(action, 3)}")
        print(f"  Δx: {delta_x:.4f}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {done}")
        print("-" * 40)

        if done:
            print("Episode ended early, resetting environment...")
            obs = env.reset()
            xpos_before = sim.data.qpos[0]

if __name__ == "__main__":
    main()
