


import os

import numpy as np
from learning_to_adapt.envs.mujoco_env import MujocoEnv
from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.logger import logger



# MujocoEnv handles loading of xml models, resetting the simulation, and setting up action/observation spaces
# Serializable makes  environment state and constructor parameters pickle-able
class RoverEnv(MujocoEnv, Serializable):

    def __init__(self, task=None, reset_every_episode=False):
        # Makes the class serializable for logging / saving
        Serializable.quick_init(self, locals())

        # Store task type (can be used later for perturbations)
        self.task = None if task == 'None' else task

        # Flags for episodic resets and first-time init
        self.reset_every_episode = reset_every_episode
        self.first = True

        # Path to rover XML model
        xml_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "assets",
            "rover.xml"
        )

        # Initialize the MuJoCo environment (loads XML & sets up self.model)
        MujocoEnv.__init__(self, xml_path)

        # Simulation timestep (seconds per control step)
        self.dt = self.model.opt.timestep
        
         
        
    def get_current_obs(self):
        """
        Build the observation vector for the rover.
        This is what the controller or policy will see at each step.
        """
        
        # qpos = all positions in the simulator
        # includes: chassis position (x, y, z), chassis orientation (quaternion), and all joint positions (wheel rotations)
        qpos = self.model.data.qpos.flat.copy()
        
        # qvel = all velocities in the simulator
        # includes: chassis linear velocity (vx, vy, vz), chassis angular velocity, and all joint angular velocities
        qvel = self.model.data.qvel.flat.copy()        
        
        lin_vel = qvel[0:3] # Chassis linear velocity (m/s) -> forward/backward and sideways speed
        ang_vel = qvel[3:6] # Chassis angular velocity (rad/s) -> how fast it's rotating around each axis
        orientation = qpos[3:7] # Chassis orientation (quaternion format) -> direction it's facing + tilt
        wheel_pos = qpos[7:] # Rotation angles of each wheel joint (radians)
        wheel_vel = qvel[6:] # Rotation speeds of each wheel joint (radians/sec)
    
        # Combine everything into one flat array and return
        return np.concatenate([lin_vel, ang_vel, orientation, wheel_pos, wheel_vel,])

        
    def step(self, action):
        """
        Apply an action to the rover, step the simulation, and calculate reward.

        Phase 1 shaping:
        reward = forward_velocity
                - |sideways_velocity|
                - small control_cost
        """
        
        xpos_before = self.get_body_com("rover")[0] # Position of the chassis COM along x before step
        self.forward_dynamics(action) # Advance the simulation by one step
        xpos_after = self.get_body_com("rover")[0] # Position of the chassis COM along x after step
        forward_vel = (xpos_after - xpos_before) / self.dt  # Forward velocity (m/s)
        sideways_vel = self.get_body_comvel("rover")[1] # Sideways velocity (m/s)
        ctrl_cost = 0.05 * np.sum(np.square(action)) # Control cost penalty
        reward = forward_vel - np.abs(sideways_vel) - ctrl_cost # Reward logic (duplicated from reward())
        obs = self.get_current_obs() # New observation after step
        done = False # No termination condition for now
        info = {} # Logging info

        return obs, reward, done, info
    
    
    
    
    def reward(self, obs, action, next_obs):
        """
        Vectorised reward function for batch processing.
        Assumes get_current_obs() layout:
        0:3 = chassis linear velocity (vx, vy, vz)
        3:6 = chassis angular velocity
        6:10 = chassis orientation quaternion
        10:14 = wheel joint positions
        14:18 = wheel joint velocities
        """
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]

        forward_vel = next_obs[:, 0] # Forward velocity (vx) from next_obs
        sideways_vel = next_obs[:, 1] # Sideways velocity (vy) from next_obs
        ctrl_cost = 0.05 * np.sum(np.square(action), axis=1)  # Control cost penalty
        reward = forward_vel - np.abs(sideways_vel) - ctrl_cost # Total reward (same as step())

        return reward
    
    
    def reset_model(self):
        """
        Reset the rover to its initial position/velocity with small random noise.
        """
        # Add tiny Gaussian noise to positions and velocities
        qpos = self.init_qpos + np.random.normal(scale=0.01, size=self.init_qpos.shape)
        qvel = self.init_qvel + np.random.normal(scale=0.1, size=self.init_qvel.shape)

        # Write state back into the simulator
        self.set_state(qpos, qvel)

        # Return initial observation
        return self.get_current_obs()
    
    
    def reset_mujoco(self, init_state=None):
        super(RoverEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False
            
            
    def reset_task(self, value=None):
        pass


    def log_diagnostics(self, paths, prefix=''):
        # Index 0 in obs = forward velocity, but for distance we need chassis x-position.
        # If x-position isn't in obs, use COM from mujoco directly in step logging.
        forward_progresses = [
            path["observations"][-1][0] - path["observations"][0][0]
            for path in paths
        ]
        logger.logkv(prefix + 'AverageForwardProgress', np.mean(forward_progresses))
        logger.logkv(prefix + 'MaxForwardProgress', np.max(forward_progresses))
        logger.logkv(prefix + 'MinForwardProgress', np.min(forward_progresses))
        logger.logkv(prefix + 'StdForwardProgress', np.std(forward_progresses))