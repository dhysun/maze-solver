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