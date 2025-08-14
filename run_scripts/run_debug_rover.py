from learning_to_adapt.envs import RoverEnv
from learning_to_adapt.envs.normalized_env import normalize
import numpy as np

def unwrap_env(env):
    """Recursively unwrap env to get to the base MuJoCo env."""
    current = env
    while hasattr(current, "_wrapped_env"):
        current = current._wrapped_env
    return current

def main():
    # Create env exactly like training scripts
    env = normalize(RoverEnv(reset_every_episode=True, task=None))
    base_env = unwrap_env(env)  # Get down to RoverEnv

    # Ensure we have MuJoCo sim access
    sim = getattr(base_env, "sim", None)
    if sim is None:
        raise RuntimeError("Base environment has no sim attribute!")

    obs = env.reset()
    print("Initial observation:", obs)

    # Run random actions for debug
    for step in range(10):
        action = np.random.uniform(-1, 1, env.action_space.shape)
        xpos_before = sim.data.qpos[0]  # Assuming first qpos is x position
        obs, reward, done, _ = env.step(action)
        xpos_after = sim.data.qpos[0]
        dx = xpos_after - xpos_before
        print(f"Step {step}: action={action}, Δx={dx:.4f}, reward={reward:.4f}")
        if done:
            print("Episode finished early.")
            break

    env.close()

if __name__ == "__main__":
    main()
