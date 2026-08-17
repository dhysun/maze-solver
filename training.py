# Training uses inverted pendulum training example as a reference:
# https://docs.pytorch.org/rl/stable/tutorials/coding_ppo.html

# imports

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing, nn

from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

import os
from maze_env import MazeEnv

# hyperparameters

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256
lr = 3e-4
max_grad_norm = 1.0

# data collection parameters

frames_per_batch = 1000
total_frames = 10_000

# ppo parameters

sub_batch_size = 64
num_epochs = 10
clip_epsilon = 0.2
gamma = 0.99
lmbda = 0.95
entropy_eps = 2e-3

view_radius = 3
min_cells, max_cells = 3, 10
max_steps = 400

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training outputs")
checkpoint_path = os.path.join(output_dir, "maze_policy.pt")
plot_path = os.path.join(output_dir, "maze_training_curves.png")

# environment

base_env = MazeEnv(min_cells = min_cells, max_cells = max_cells,
                    view_radius = view_radius, max_steps = max_steps)

gym_env = GymWrapper(base_env, device = device, categorical_action_encoding = True)

env = TransformedEnv(gym_env, DoubleToFloat())

rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)

# policy

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device = device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device = device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device = device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device = device),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys = ["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module = policy_module,
    spec = env.action_spec,
    in_keys = ["loc", "scale"],
    distribution_class = TanhNormal,
    distribution_kwargs = {
        "low": env.action_spec_unbatched.space.low,
        "high": env.action_spec_unbatched.space.high,
    },
    return_log_prob = True,
)

# value network

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device = device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device = device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device = device),
    nn.Tanh(),
    nn.LazyLinear(1, device = device),
)

value_module = ValueOperator(
    module = value_net,
    in_keys = ["observation"],
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

# data collector

collector = Collector(
    env,
    policy_module,
    frames_per_batch = frames_per_batch,
    total_frames = total_frames,
    split_trajs = False,
    device = device,
)

# replay buffer

replay_buffer = ReplayBuffer(
    storage = LazyTensorStorage(max_size = frames_per_batch),
    sampler = SamplerWithoutReplacement(),
)

# loss function

advantage_module = GAE(
    gamma = gamma, lmbda = lmbda, value_network = value_module, average_gae = True
)

loss_module = ClipPPOLoss(
    actor_network = policy_module,
    critic_network = value_module,
    clip_epsilon = clip_epsilon,
    entropy_bonus = bool(entropy_eps),
    entropy_coeff = entropy_eps,
    critic_coeff = 1.0,
    loss_critic_type = "smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

# training loop

logs = defaultdict(list)
pbar = tqdm(total = total_frames)
eval_str = ""

for i, tensordict_data in enumerate(collector):
    for _ in range(num_epochs):
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward = {logs['reward'][-1]: 4.4f} (init = {logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    scheduler.step()

# results

plt.figure(figsize = (10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()