import gymnasium as gym
import random

from gymnasium import spaces
import numpy as np

class MazeEnv(gym.Env):

    def __init__(self, min_cells = 4, max_cells = 8, view_radius = 2, 
                 max_steps = 300, seed: int | None = None):
        super().__init__()
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.view_radius = view_radius
        self.max_steps = max_steps
        
        self._rng = random.Random(seed)

        view_len = (2 * view_radius) + 1
        observation_dim = (view_len * view_len) + 2

        self.observation_space = spaces.Box(low = -1.0, high = 1.0, 
                                            shape = (observation_dim,), 
                                            dtype = np.float32)
        self.action_space = spaces.Discrete(4)

        self._moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self._provided_maze = None
        self._provided_start = None
        self._provided_goal = None

