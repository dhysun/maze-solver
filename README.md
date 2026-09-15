# ASCII Maze Solver

A reinforcement learning agent (trained with PPO) which solves ASCII-generated mazes. At each state, the agent sees the surrounding positions and the direction to the goal, and chooses to move up, down, left, or right while avoiding walls. The training of the reinforcement learning agent is done with a curriculum which progresses from 7x7 to 17x17 mazes. The policy and weights are saved in maze_policy.pt. 

Test cases generated on https://www.asciiart.eu/ascii-maze-generator are located the "test mazes" folder and the outputs of solution attempts are in the "test solutions" folder.

The agent had a 93% success rate on 11x11-11x17 mazes, 70% on 13x11-13x17 mazes, and 35% on 15x15-17x17 mazes.