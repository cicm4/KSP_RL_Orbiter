"""Controlled optimizer and exploration experiments, separate from release policies."""

import hashlib
import json
import math

import pytest

torch = pytest.importorskip("torch")

from orbiter import train_generalized as training


def source_checkpoint(path):
    torch.manual_seed(71)
    model = training.VehiclePolicy(16)
    training.save_policy(path, model, training.experiment_contract("stock")[2], {})
    return model


def test_separate_gradient_clipping_preserves_actor_when_critic_is_huge():
    """Reproduce the observed coupling without assuming it explains poor flights."""
    actor = torch.nn.Parameter(torch.tensor([1.0]))
    log_std = torch.nn.Parameter(torch.tensor([1.0]))
    critic = torch.nn.Parameter(torch.tensor([1.0]))
    actor.grad = torch.tensor([0.3])
    log_std.grad = torch.tensor([0.4])
    critic.grad = torch.tensor([1e6])
    training.clip_ppo_gradients(
        [actor, log_std], [critic], optimizer="joint"
    )
    assert actor.grad.item() < 1e-6
    actor.grad = torch.tensor([0.3])
    log_std.grad = torch.tensor([0.4])
    critic.grad = torch.tensor([1e6])
    norm, actor_norm, critic_norm = training.clip_ppo_gradients(
        [actor, log_std], [critic], optimizer="separate",
        actor_max_grad_norm=0.25, critic_max_grad_norm=0.1,
    )
    assert actor.grad.item() == pytest.approx(0.15, rel=1e-5)
    assert log_std.grad.item() == pytest.approx(0.2, rel=1e-5)
    assert critic.grad.item() == pytest.approx(0.1, rel=1e-5)
    assert actor_norm.item() == pytest.approx(0.5)
    assert critic_norm.item() == pytest.approx(1e6)
    assert norm.item() == pytest.approx(math.hypot(0.5, 1e6))


def test_explicit_std_reset_preserves_actor_and_same_seed_critic(tmp_path):
    source = source_checkpoint(tmp_path / "parent.pt")
    contract = training.experiment_contract("stock", "aero", deadline_mode="terminal")[2]
    baseline, _ = training.initialize_actor_policy(
        tmp_path / "parent.pt", contract, seed=601
    )
    reset, _ = training.initialize_actor_policy(
        tmp_path / "parent.pt", contract, seed=601, initial_action_std=0.25
    )
    for key, value in baseline.actor.state_dict().items():
        torch.testing.assert_close(reset.actor.state_dict()[key], value, rtol=0, atol=0)
    for key, value in baseline.critic.state_dict().items():
        torch.testing.assert_close(reset.critic.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(baseline.log_std, source.log_std, rtol=0, atol=0)
    torch.testing.assert_close(reset.log_std.exp(), torch.full((2,), 0.25))


def test_optin_optimizer_contract_roundtrip_and_tamper_rejection(tmp_path):
    variant = training.ppo_optimizer_configuration(
        optimizer="separate", learning_rate=1e-5,
        actor_learning_rate=3e-6, critic_learning_rate=3e-4,
        actor_max_grad_norm=0.5, critic_max_grad_norm=1.0,
        entropy_coefficient=0.001,
    )
    contract = training.experiment_contract(
        "stock", "aero", deadline_mode="terminal", ppo_variant=variant
    )[2]
    assert contract["task"] == training.AERO_PPO_VARIANT_TASK
    assert contract["ppo_variant"] == variant
    path = tmp_path / "variant.pt"
    training.save_policy(path, training.VehiclePolicy(16, observations=56), contract, {})
    _, record = training.load_policy(path)
    assert record["contract"] == contract
    for change in (
        {"unknown_option": True},
        {"entropy_coefficient": float("nan")},
        {"actor_learning_rate": -1.0},
        {"entropy_estimator": "latent_normal"},
    ):
        altered = torch.load(path, weights_only=True)
        altered["contract"]["ppo_variant"].update(change)
        bad = tmp_path / "tampered.pt"
        torch.save(altered, bad)
        with pytest.raises(ValueError, match="variant"):
            training.load_policy(bad)


def test_squashed_entropy_gradient_discourages_saturation():
    torch.manual_seed(12)
    mean = torch.tensor([[5.0, -5.0]], requires_grad=True)
    std = torch.full_like(mean, 0.2)
    entropy = training.squashed_normal_entropy(torch.distributions.Normal(mean, std))
    (-entropy).backward()
    assert mean.grad[0, 0] > 0
    assert mean.grad[0, 1] < 0
    assert torch.isfinite(entropy)


@pytest.mark.parametrize("extra", [
    ["--ppo-optimizer", "separate", "--actor-learning-rate", "nan"],
    ["--ppo-optimizer", "separate", "--critic-learning-rate", "0"],
    ["--ppo-optimizer", "separate", "--actor-max-grad-norm", "inf"],
    ["--ppo-optimizer", "separate", "--critic-max-grad-norm", "-1"],
    ["--actor-learning-rate", "1e-5"],
    ["--actor-max-grad-norm", "0.5"],
    ["--entropy-coefficient", "-1"],
    ["--entropy-coefficient", "nan"],
    ["--initial-action-std", "0"],
    ["--initial-action-std", "nan"],
    ["--initial-action-std", "0.2"],
])
def test_invalid_variant_cli_fails_before_output(tmp_path, extra):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit):
        training.main(["--run-dir", str(output), "--device", "cpu", *extra])
    assert not output.exists()


def test_historical_default_ppo_numeric_regression():
    """Recorded before adding variants; zero entropy must consume no extra RNG."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(321)
        model = training.VehiclePolicy(4)
        env = training.BatchedAscent(4, seed=321, max_time=4)
        events = []
        training.ppo_train(
            model, env, updates=1, horizon=2, batch_size=8, learning_rate=1e-5,
            target_kl=0.02, log=lambda _event, **row: events.append(row),
            selected_callback=lambda _: None,
        )
    finally:
        torch.set_num_threads(previous_threads)
    torch.testing.assert_close(
        model.log_std, torch.tensor([-3.000040054321289, -2.999959945678711]),
        rtol=0, atol=2e-7,
    )
    torch.testing.assert_close(
        model.actor[-1].bias, torch.tensor([0.2898525595664978, -0.28736114501953125]),
        rtol=0, atol=2e-7,
    )
    torch.testing.assert_close(
        model.critic[-1].bias, torch.tensor([-0.13798314332962036]), rtol=0, atol=2e-7
    )
    assert events[0]["optimization_minibatches"] == 4
    assert events[0]["mean_squashed_entropy"] is None


def test_separate_adam_actor_clip_is_not_scaled_by_large_critic(monkeypatch):
    original_mse = torch.nn.functional.mse_loss
    original_step = torch.optim.Adam.step
    actor_norms = {}
    log_std_membership = {}
    rates = {}
    current = None
    current_model = None

    def huge_mse(*args, **kwargs):
        return 1e12 * original_mse(*args, **kwargs)

    def observed_step(optimizer, *args, **kwargs):
        actor_ids = {id(p) for p in current_model.actor.parameters()} | {id(current_model.log_std)}
        params = [p for group in optimizer.param_groups for p in group["params"]]
        members = [p for p in params if id(p) in actor_ids and p.grad is not None]
        if members and current not in actor_norms:
            actor_norms[current] = float(torch.linalg.vector_norm(torch.stack([
                p.grad.norm() for p in members
            ])))
            log_std_membership[current] = any(p is current_model.log_std for p in params)
        rates.setdefault(current, []).append(optimizer.param_groups[0]["lr"])
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "mse_loss", huge_mse)
    monkeypatch.setattr(torch.optim.Adam, "step", observed_step)
    for current in ("joint", "separate"):
        torch.manual_seed(33)
        current_model = training.VehiclePolicy(8)
        env = training.BatchedAscent(4, seed=33, max_time=4)
        extra = {} if current == "joint" else dict(
            actor_learning_rate=3e-6, critic_learning_rate=3e-4,
            actor_max_grad_norm=0.25, critic_max_grad_norm=0.1,
        )
        training.ppo_train(
            current_model, env, updates=1, horizon=2, batch_size=8,
            learning_rate=1e-5, ppo_optimizer=current, target_kl=1e-8,
            log=lambda *_a, **_kw: None, selected_callback=lambda _: None, **extra,
        )
    assert actor_norms["joint"] < 1e-8
    assert actor_norms["separate"] == pytest.approx(0.25, rel=1e-5)
    assert all(log_std_membership.values())
    assert set(rates["joint"]) == {1e-5}
    assert set(rates["separate"]) == {3e-6, 3e-4}


def test_extra_critic_fits_continue_after_kl_without_mutating_actor():
    models, events = [], []
    for extra_updates in (0, 3):
        torch.manual_seed(44)
        model = training.VehiclePolicy(8)
        env = training.BatchedAscent(4, seed=44, max_time=4)
        event = []
        training.ppo_train(
            model, env, updates=1, horizon=2, batch_size=8,
            learning_rate=1e-3, ppo_optimizer="separate", target_kl=1e-8,
            critic_extra_updates=extra_updates,
            log=lambda _event, **row: event.append(row), selected_callback=lambda _: None,
        )
        models.append(model)
        events.append(event[0])
    assert all(row["kl_early_stop"] for row in events)
    assert all(row["cumulative_actor_optimizer_steps"] == 1 for row in events)
    assert events[0]["cumulative_critic_optimizer_steps"] == 1
    assert events[1]["cumulative_critic_optimizer_steps"] == 4
    assert events[1]["extra_critic_loss"] > 0
    assert math.isfinite(events[1]["extra_critic_gradient_norm_before_clip"])
    for first, second in zip(models[0].actor.parameters(), models[1].actor.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(models[0].log_std, models[1].log_std, rtol=0, atol=0)
    assert any(not torch.equal(a, b) for a, b in zip(
        models[0].critic.parameters(), models[1].critic.parameters()
    ))


def test_variant_main_records_reset_provenance_and_rejects_silent_warmstart(tmp_path, monkeypatch):
    source_checkpoint(tmp_path / "parent.pt")
    selected_calls = 0

    def evaluate(_model, *, count, **_kwargs):
        nonlocal selected_calls
        selected_calls += 1
        return dict(episodes=count, successes=1, success_rate=1 / count,
                    success_ci95=[0, 1], failure_counts={})

    def fake_ppo(_model, _env, *, selected_callback, **kwargs):
        assert kwargs["ppo_optimizer"] == "separate"
        assert kwargs["critic_extra_updates"] == 2
        assert kwargs["entropy_coefficient"] == 0.001
        selected_callback(8)
        return 8

    monkeypatch.setattr(training, "evaluate", evaluate)
    monkeypatch.setattr(training, "ppo_train", fake_ppo)
    output = tmp_path / "variant"
    args = [
        "--run-dir", str(output), "--device", "cpu", "--physics", "aero",
        "--curriculum", "stock", "--actor-from", str(tmp_path / "parent.pt"),
        "--ppo-only", "--ppo-updates", "1", "--value-updates", "0",
        "--worlds", "4", "--eval-worlds", "8", "--evaluation-suite", "ablation",
        "--deadline-mode", "terminal", "--ppo-optimizer", "separate",
        "--actor-learning-rate", "3e-6", "--critic-learning-rate", "3e-4",
        "--initial-action-std", "0.2", "--entropy-coefficient", "0.001",
        "--critic-extra-updates", "2",
    ]
    training.main(args)
    _, record = training.load_policy(output / "best.pt")
    init = record["ancestry"]["initialization"]
    assert init["log_std_copied"] is False
    assert init["log_std_reset"] is True
    assert init["initial_action_std"] == 0.2
    assert init["parent_action_std"] == pytest.approx([math.exp(-3)] * 2)
    assert record["ancestry"]["ppo_optimization"]["variant"] == record["contract"]["ppo_variant"]
    assert record["contract"]["ppo_variant"]["critic_extra_updates"] == 2
    wrong_output = tmp_path / "wrong-contract"
    with pytest.raises(SystemExit):
        training.main([
            "--run-dir", str(wrong_output), "--device", "cpu", "--physics", "aero",
            "--curriculum", "stock", "--warm-start", str(output / "best.pt"),
            "--ppo-only", "--deadline-mode", "terminal",
        ])
    assert not wrong_output.exists()


@pytest.mark.parametrize("extra", [
    ["--critic-extra-updates", "1"],
    ["--ppo-optimizer", "separate", "--critic-extra-updates", "-1"],
    ["--ppo-optimizer", "separate", "--critic-extra-updates", "1.5"],
])
def test_invalid_extra_critic_updates_fail_before_output(tmp_path, extra):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit):
        training.main(["--run-dir", str(output), "--device", "cpu", *extra])
    assert not output.exists()


@pytest.mark.parametrize("value", [1e-300, 1e300])
def test_std_reset_rejects_unrepresentable_policy_distribution(tmp_path, value):
    source_checkpoint(tmp_path / "parent.pt")
    with pytest.raises(ValueError, match="representable"):
        training.initialize_actor_policy(
            tmp_path / "parent.pt", training.experiment_contract("stock")[2],
            seed=601, initial_action_std=value,
        )


def test_entropy_only_joint_variant_is_distinct_and_rejects_bad_objective(tmp_path):
    variant = training.ppo_optimizer_configuration(entropy_coefficient=0.001)
    contract = training.experiment_contract("stock", ppo_variant=variant)[2]
    assert contract["task"] == training.PPO_VARIANT_TASK
    assert "objective" not in contract
    path = tmp_path / "joint-entropy.pt"
    training.save_policy(path, training.VehiclePolicy(16), contract, {})
    training.load_policy(path)
    record = torch.load(path, weights_only=True)
    record["contract"]["objective"] = None
    torch.save(record, path)
    with pytest.raises(ValueError, match="variant"):
        training.load_policy(path)
