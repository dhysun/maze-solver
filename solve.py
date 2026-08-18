import torch
from torch import nn

import sys
import os

from maze_env import MazeEnv, parse_ascii_maze

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

def main():
    # check if maze was provided
    if len(sys.argv) < 2:
        print("maze not provided")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        maze_text = f.read()
    
    maze, start, goal = parse_ascii_maze(maze_text)
    env = MazeEnv(view_radius = VIEW_RADIUS, max_steps = MAX_STEPS)
    env.set_provided_maze(maze, start, goal)

    actor_net = build_actor(num_actions = 4)

    dummy_obs, _ = env.reset()
    actor_net(torch.as_tensor(dummy_obs).unsqueeze(0))

    checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "training outputs", "maze_policy.pt"
    )

    actor_net.load_state_dict(torch.load(checkpoint_path, map_location = "cpu"))
    actor_net.eval()

    obs, _ = env.reset()
    path = [env.pos]
    solved = False
    stall_counter = 0
    STALL_LIMIT = 6

    with torch.no_grad():

        for _ in range(MAX_STEPS):
            logits = actor_net(torch.as_tensor(obs).unsqueeze(0))
            
            if stall_counter < STALL_LIMIT:
                action = int(torch.argmax(logits, dim = -1).item())
            else:
                dist = torch.distributions.Categorical(logits = logits)
                action = int(dist.sample().item())
            
            prev_pos = env.pos
            obs, reward, terminated, truncated, _ = env.step(action)
            path.append(env.pos)

            if env.pos != prev_pos:
                stall_counter = 0
            else:
                stall_counter += 1
            
            if terminated:
                solved = True
                break
            
            if truncated:
                break
    
    if solved:
        print(f"Solved in {len(path) - 1} steps\n")
    else:
        print(f"No solution found after {len(path) - 1} steps\n")

    print_maze_solution(maze, path, start, goal)

if __name__ == "__main__":
    main()