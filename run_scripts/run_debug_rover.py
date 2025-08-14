import numpy as np
from learning_to_adapt.envs import rover_env
from learning_to_adapt.envs.normalized_env import normalize

def main():
    # Create RoverEnv in same style as run_baseline.py
    env = normalize(rover_env.RoverEnv(reset_every_episode=True))

    obs = env.reset()
    print("Initial observation:", obs)

    for step in range(10):
        # Random actions for 4 wheels in [-1, 1]
        action = np.random.uniform(low=-1, high=1, size=env.action_space.shape)

        obs, reward, done, info = env.step(action)

        # Access MuJoCo sim directly
        sim = env.wrapped_env.sim

        # Joint positions for all wheels
        wheel_pos = {
            name: sim.data.qpos[sim.model.get_joint_qpos_addr(name)]
            for name in ["wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint"]
        }

        # Base/root body position (x, y, z)
        rover_pos = sim.data.qpos[0:3]

        print(f"\nStep {step + 1}")
        print("  Action:", np.round(action, 3))
        print("  Wheel positions:", {k: round(v, 4) for k, v in wheel_pos.items()})
        print("  Rover position:", np.round(rover_pos, 4))
        print("  Reward:", round(reward, 4))

        if done:
            print("Episode ended early.")
            break

    env.close()

if __name__ == "__main__":
    main()
