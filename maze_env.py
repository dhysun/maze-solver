import gymnasium as gym
import random

from gymnasium import spaces
import numpy as np

def create_maze(cells_width, cells_height, rng: random.Random) -> np.ndarray:
    height, width = (2 * cells_height) + 1, (2 * cells_width) + 1
    
    # initialize maze as all 1's
    maze = np.ones((height, width), dtype = np.uint8)

    # initialize visited cells as all false
    visited_cells = np.zeros((cells_height, cells_width), dtype = bool)

    # converts cell coords to maze coords
    def to_maze_coord(cell_y, cell_x):
        return (2 * cell_y) + 1, (2 * cell_x) + 1
    
    stack = [(0, 0)]
    visited_cells[0, 0] = True

    maze_y, maze_x = to_maze_coord(0, 0)
    maze[maze_y, maze_x] = 0

    while stack:
        cell_y, cell_x = stack[-1]
        nbors = []

        # append potential neighbors which are in bounds
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nbor_y, nbor_x = cell_y + dy, cell_x + dx
            if (0 <= nbor_y < cells_height and 0 <= nbor_x < cells_width 
                    and not visited_cells[nbor_y, nbor_x]):
                nbors.append((nbor_y, nbor_x, dy, dx))
        
        if nbors:
            # randomly pick one of the neighbors
            nbor_y, nbor_x, dy, dx = rng.choice(nbors)

            # convert cell to maze coordinates and set the wall between
            # the current cell and the neighbor to a path by changing the
            # value from 1 (representing wall) to 0 (representing path)
            maze_y, maze_x = to_maze_coord(cell_y, cell_x)
            maze[maze_y + dy, maze_x + dx] = 0

            # do the same for the neighbor cell
            nbor_maze_y, nbor_maze_x = to_maze_coord(nbor_y, nbor_x)
            maze[nbor_maze_y, nbor_maze_x] = 0

            visited_cells[nbor_y, nbor_x] = True
            stack.append((nbor_y, nbor_x))
        else:
            stack.pop()
    
    return maze

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

        # observation_space is Space object for all valid observations
        # observations are 1d float array of length observation_dim and
        # values between -1.0 and 1.0
        self.observation_space = spaces.Box(low = -1.0, high = 1.0, 
                                            shape = (observation_dim,), 
                                            dtype = np.float32)

        # action_space is Space object for all valid actions
        # action space is {0, 1, 2, 3}
        self.action_space = spaces.Discrete(4)

        # possible moves agent can make (up, down, left, right)
        self._moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self._provided_maze = None
        self._provided_start = None
        self._provided_goal = None
    
    def set_provided_maze(self, maze: np.ndarray, start: tuple, goal: tuple):
        self._provided_maze = maze
        self._provided_start = start
        self._provided_goal = goal

    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)
        
        # get random size for maze
        cells_w = self._rng.randint(self.min_cells, self.max_cells)
        cells_h = self._rng.randint(self.min_cells, self.max_cells)
        
        # create the maze
        self.maze = create_maze(cells_w, cells_h, self._rng)

        h, w = self.maze.shape
        self.pos = (1, 1) # top left cell
        self.goal = (h - 2, w - 2) # bottom-right cell
        self.steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}