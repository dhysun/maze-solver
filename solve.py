from torch import nn

NUM_CELLS = 256
VIEW_RADIUS = 3
MAX_STEPS = 400

def build_actor(num_actions):
    return nn.Sequential(
        nn.LazyLinear(NUM_CELLS),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS),
        nn.Tanh(),
        nn.LazyLinear(num_actions)
    )

def print_maze_solution(maze, path, start, goal):
    res = []

    for y in range(maze.shape[0]):
        row = []

        for x in range(maze.shape[1]):
            if maze[y, x] == 1:
                row.append("#")
            else:
                row.append(" ")
        
        res.append(row)

    for (y, x) in path:
        res[y][x] = "."
    
    start_y, start_x = start
    goal_y, goal_x = goal

    res[start_y][start_x] = "S"
    res[goal_y][goal_x] = "E"

    for row in res:
        print("".join(row))