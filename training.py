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
from torchrl.envs.libs.gym import GymWrapper

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

frames_per_batch = 2000
total_frames = 5_000_000

# ppo parameters

sub_batch_size = 64
num_epochs = 10
clip_epsilon = 0.2
gamma = 0.99
lmbda = 0.95

entropy_eps = 2e-3
entropy_eps_final = 1e-4

view_radius = 3
min_cells, max_cells = 3, 8
max_steps = 300

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training outputs")
checkpoint_path = os.path.join(output_dir, "maze_policy.pt")
plot_path = os.path.join(output_dir, "maze_training_curves.png")
os.makedirs(output_dir, exist_ok = True)

# environment

base_env = MazeEnv(min_cells = min_cells, max_cells = max_cells,
                    view_radius = view_radius, max_steps = max_steps)

gym_env = GymWrapper(base_env, device = device, categorical_action_encoding = True)

env = TransformedEnv(gym_env, DoubleToFloat())

rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)

# policy

num_actions = env.action_spec.space.n

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device = device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device = device),
    nn.Tanh(),
    nn.LazyLinear(num_actions, device = device)
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys = ["logits"]
)

policy_module = ProbabilisticActor(
    module = policy_module,
    spec = env.action_spec,
    in_keys = ["logits"],
    distribution_class = torch.distributions.Categorical,
    return_log_prob = True
)

# value network

value_net = nn.Sequential(
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
    optim, total_frames // frames_per_batch, lr * 0.1
)

# training loop

logs = defaultdict(list)
pbar = tqdm(total = total_frames)
eval_str = ""

for i, tensordict_data in enumerate(collector):

    advantage_module(tensordict_data)
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view.cpu())

    for _ in range(num_epochs):
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
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

    frac = pbar.n / total_frames
    if frac < 0.25:
        base_env.min_cells, base_env.max_cells = 3, 4
    elif frac < 0.45:
        base_env.min_cells, base_env.max_cells = 3, 6
    else:
        base_env.min_cells, base_env.max_cells = min_cells, max_cells

    curriculum_str = f"maze cells: {base_env.min_cells}-{base_env.max_cells}"

    current_entropy = entropy_eps * (1 - frac) + entropy_eps_final * frac
    loss_module.entropy_coeff = current_entropy
    entropy_str = f"entropy: {current_entropy:.5f}"

    if i % 5 == 0:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(max_steps, policy_module)
            success = bool(eval_rollout["next", "terminated"].any().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval success"].append(1.0 if success else 0.0)
            eval_str = (
                f"eval return: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(solved: {success})"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str, curriculum_str, entropy_str]))
    scheduler.step()

torch.save(actor_net.state_dict(), checkpoint_path)
print(f"saved policy weights to {checkpoint_path}")

# results

plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(1, 2, 2)
plt.plot(logs["eval success"])
plt.title("eval solved?")
plt.savefig(plot_path)
print(f"saved training curves to {plot_path}")