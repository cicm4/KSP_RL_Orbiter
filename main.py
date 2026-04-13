import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchrl.modules import ProbabilisticActor, SafeModule, ValueOperator
from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
from torchrl.objectives import group_optimizers
from torchrl.objectives.sac import SACLoss
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from ksp_state import KSPState
from qvn_model import QVNModel


N_OBS = 10
N_ACTIONS = 3
HIDDEN = 256
DEFAULT_DATA_DIR = Path("data")
DEFAULT_CHECKPOINT_NAME = "checkpoint_latest.pt"


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def checkpoint_episode(path: Path) -> int:
    stem = path.stem
    if stem.startswith("checkpoint_ep"):
        suffix = stem.removeprefix("checkpoint_ep")
        if suffix.isdigit():
            return int(suffix)
    return -1


def find_latest_checkpoint(data_dir: Path) -> Path:
    latest_path = data_dir / DEFAULT_CHECKPOINT_NAME
    if latest_path.exists():
        return latest_path

    episodic = sorted(
        data_dir.glob("checkpoint_ep*.pt"),
        key=checkpoint_episode,
    )
    if episodic:
        return episodic[-1]

    raise FileNotFoundError(
        f"No checkpoints found in {data_dir}. Expected {latest_path} or checkpoint_ep*.pt."
    )


def build_env(target_orbit_altitude: int, step_interval: float, max_steps: int) -> KSPState:
    return KSPState(
        target_orbit_altitude=target_orbit_altitude,
        step_interval=step_interval,
        max_steps=max_steps,
    )


def build_actor(env: KSPState, device: torch.device) -> ProbabilisticActor:
    actor_net = nn.Sequential(
        nn.Linear(N_OBS, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 2 * N_ACTIONS),
        NormalParamExtractor(),
    )

    actor_module = SafeModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        spec=env.action_spec,
        distribution_class=TanhNormal,
    ).to(device)
    return actor


def build_qvalue(device: torch.device) -> ValueOperator:
    return ValueOperator(
        module=QVNModel(N_OBS, N_ACTIONS, HIDDEN),
        in_keys=["observation", "action"],
    ).to(device)


def build_loss_module(actor: ProbabilisticActor, qvalue: ValueOperator, device: torch.device):
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        alpha_init=0.2,
        target_entropy="auto",
        loss_function="smooth_l1",
    )
    loss_module.to(device)
    return loss_module


def build_optimizer(loss_module):
    optimizer_actor = torch.optim.Adam(
        loss_module.actor_network_params.values(True, True), lr=3e-4
    )
    optimizer_critic = torch.optim.Adam(
        loss_module.qvalue_network_params.values(True, True), lr=3e-4
    )
    optimizer_alpha = torch.optim.Adam([loss_module.log_alpha], lr=3e-4)
    return group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)


def load_latest_model(
    data_dir: Path,
    env: KSPState,
    device: torch.device,
):
    # Checkpoints store the full SAC loss module, so rebuild that graph first.
    actor = build_actor(env, device)
    qvalue = build_qvalue(device)
    loss_module = build_loss_module(actor, qvalue, device)
    optimizer = build_optimizer(loss_module)

    checkpoint_path = find_latest_checkpoint(data_dir)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    incompatible = loss_module.load_state_dict(
        checkpoint["loss_module"], strict=False
    )
    if incompatible.missing_keys:
        print(f"Missing checkpoint keys ignored: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"Unexpected checkpoint keys ignored: {incompatible.unexpected_keys}")

    if "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            pass

    actor.eval()
    qvalue.eval()

    start_episode = checkpoint.get("episode", -1) + 1
    total_steps = checkpoint.get("total_steps", 0)
    return actor, checkpoint_path, start_episode, total_steps


def run_use_scenario(
    actor: ProbabilisticActor,
    env: KSPState,
    device: torch.device,
):
    td = env.reset()
    episode_return = 0.0

    while True:
        td_device = td.to(device)
        # Use the deterministic policy path for deployment.
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            td_device = actor(td_device)

        td["action"] = td_device["action"].cpu()
        next_td = env.step(td)

        reward = next_td["next", "reward"].item()
        episode_return += reward

        if next_td["next", "done"].item():
            break

        td = step_mdp(next_td)

    return episode_return, env.get_step_info()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the latest trained policy and run it in a normal KSP use scenario."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing checkpoint_latest.pt and checkpoint_ep*.pt",
    )
    parser.add_argument(
        "--target-orbit-altitude",
        type=int,
        default=80_000,
        help="Target orbit altitude in meters.",
    )
    parser.add_argument(
        "--step-interval",
        type=float,
        default=0.5,
        help="Environment step interval in seconds.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum steps for the scenario.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    env = build_env(
        target_orbit_altitude=args.target_orbit_altitude,
        step_interval=args.step_interval,
        max_steps=args.max_steps,
    )

    try:
        actor, checkpoint_path, start_episode, total_steps = load_latest_model(
            data_dir=args.data_dir,
            env=env,
            device=device,
        )
        print(
            f"Loaded policy from {checkpoint_path} "
            f"(next episode {start_episode}, total_steps {total_steps})"
        )

        episode_return, final_info = run_use_scenario(actor, env, device)
        print(
            "Scenario complete | "
            f"success: {final_info.get('success', False)} | "
            f"return: {episode_return:.2f} | "
            f"reason: {final_info.get('termination_reason', '')} | "
            f"apoapsis: {final_info.get('apoapsis', float('nan')):.1f} | "
            f"periapsis: {final_info.get('periapsis', float('nan')):.1f} | "
            f"fuel_remaining: {final_info.get('fuel_remaining', float('nan')):.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
