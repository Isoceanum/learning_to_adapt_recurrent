from learning_to_adapt.envs.normalized_env import normalize
from learning_to_adapt.envs import RoverEnv
import numpy as np

def unwrap_env(env):
    """Recursively unwrap to get to the base env with .sim."""
    current_env = env
    while hasattr(current_env, 'env'):
        current_env = current_env.env
    return current_env

def main():
    env = normalize(RoverEnv(reset_every_episode=True, task=None))
    
    # Unwrap to base MuJoCo env to access sim
    base_env = unwrap_env(env)
    sim = base_env.sim
    
    obs = env.reset()
    print("Initial observation:", obs)
    
    prev_x = sim.data.qpos[0]
    
    for step in range(10):
        action = np.random.uniform(-1, 1, size=env.action_space.shape)
        obs, reward, done, _ = env.step(action)
        
        new_x = sim.data.qpos[0]
        delta_x = new_x - prev_x
        prev_x = new_x
        
        print(f"Step {step}: Action={action}, Δx={delta_x:.4f}, Reward={reward:.4f}")
        
        if done:
            print("Episode finished early.")
            break

if __name__ == "__main__":
    main()
