"""Deadline ablations preserve the historical time-limit objective by default."""

import pytest

torch = pytest.importorskip("torch")

from orbiter.generalized_aero_env import BatchedAeroAscent
from orbiter.generalized_env import BatchedAscent


DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA hardware required"
        ),
    ),
]
ENVIRONMENTS = [BatchedAscent, BatchedAeroAscent]


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("penalty", [0.0, -30.0])
def test_deadline_is_terminal_and_cancels_potential(environment, device, penalty):
    batch = environment(
        2,
        seed=505,
        device=device,
        max_time=0.125,
        deadline_mode="terminal",
        timeout_penalty=penalty,
    )
    previous = batch.previous_potential.clone()
    _, reward, terminated, truncated, info = batch.step(batch.teacher())
    assert bool(terminated.all())
    assert not bool(truncated.any())
    assert info["reason"].tolist() == [5, 5]
    assert info["time"].tolist() == [0.125, 0.125]
    torch.testing.assert_close(info["elapsed"], torch.full_like(previous, 0.125))
    torch.testing.assert_close(info["shaping_reward"], -previous, rtol=0, atol=0)
    torch.testing.assert_close(
        info["terminal_reward"], torch.full_like(previous, penalty)
    )
    torch.testing.assert_close(
        reward, (-previous + penalty - 0.003 * 0.125).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        info["discounted_task_return"], info["task_reward"], rtol=0, atol=0
    )
    assert info["steps"].tolist() == [1, 1]


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("device", DEVICES)
def test_physical_termination_has_priority_at_the_deadline(environment, device):
    batch = environment(
        4,
        seed=91,
        device=device,
        max_time=0.125,
        deadline_mode="terminal",
        timeout_penalty=-9.0,
    )
    radius = batch.radius + batch.target[0]
    # Success, impact, escape and final-stage fuel exhaustion at the same time
    # as the mission deadline must retain their historical reason and reward.
    batch.state[0] = torch.stack(
        (radius, radius * 0, radius * 0, (batch.mu / radius).sqrt())
    )
    batch.state[1] = torch.tensor(
        [batch.radius + 0.01, 0, -100, 0], dtype=batch.dtype, device=device
    )
    batch.state[2] = torch.tensor(
        [batch.radius + 100000, 0, 0, 5000], dtype=batch.dtype, device=device
    )
    batch.state[3] = torch.tensor(
        [batch.radius + 10000, 0, 10, 300], dtype=batch.dtype, device=device
    )
    batch.stage[3] = batch.stage_count[3] - 1
    batch.fuel[3] = 0
    batch.previous_potential.copy_(batch.potential())
    action = torch.tensor([[-1.0, 1.0]] * 4, device=device)
    _, _, terminated, truncated, info = batch.step(action)
    assert bool(terminated.all()) and not bool(truncated.any())
    assert info["reason"].tolist() == [1, 2, 3, 4]
    assert info["terminal_reward"].tolist() == [100, -30, -30, -30]


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("deadline_mode", "penalty"),
    [("truncate", 0.0), ("terminal", 0.0), ("terminal", -30.0)],
)
def test_discounted_shaping_telescopes_at_partial_control_deadline(
    environment, device, deadline_mode, penalty
):
    batch = environment(
        2,
        seed=11,
        device=device,
        max_time=5.125,
        deadline_mode=deadline_mode,
        timeout_penalty=penalty,
    )
    initial_potential = batch.potential().clone()
    discounted_shaping = batch.zeros(2)
    discounted_task = batch.zeros(2)
    elapsed = batch.zeros(2)
    for step in range(3):
        _, reward, terminated, truncated, info = batch.step(batch.teacher())
        discount = batch.gamma**step
        discounted_shaping += discount * info["shaping_reward"]
        discounted_task += discount * info["task_reward"]
        elapsed += info["elapsed"]
        torch.testing.assert_close(
            reward,
            (
                info["shaping_reward"]
                + info["terminal_reward"]
                + info["time_reward"]
            ).float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            info["discounted_task_return"], discounted_task, rtol=1e-14, atol=1e-14
        )
    assert info["steps"].tolist() == [3, 3]
    torch.testing.assert_close(elapsed, torch.full_like(elapsed, 5.125))
    assert info["reason"].tolist() == [5, 5]
    if deadline_mode == "terminal":
        assert bool(terminated.all()) and not bool(truncated.any())
        expected = -initial_potential
    else:
        assert not bool(terminated.any()) and bool(truncated.all())
        expected = -initial_potential + batch.gamma**3 * batch.potential()
    torch.testing.assert_close(discounted_shaping, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_explicit_legacy_objective_has_bitwise_same_sampling_and_rewards(environment):
    default = environment(4, seed=28, max_time=5.125)
    explicit = environment(
        4, seed=28, max_time=5.125, deadline_mode="truncate", timeout_penalty=0.0
    )
    for step in range(3):
        action = default.teacher()
        expected = default.step(action)
        actual = explicit.step(action)
        for index in range(4):
            assert torch.equal(expected[index], actual[index])
        # The default keeps its exact historical floating-point expression.
        old_reward = (
            actual[4]["shaping_reward"]
            + torch.where(actual[2], torch.where(actual[4]["success"], 100, -30), 0)
            - 0.003 * actual[4]["elapsed"]
        ).float()
        assert torch.equal(actual[1], old_reward)
        assert torch.equal(default.generator.get_state(), explicit.generator.get_state())
        if step == 1:
            mask = torch.tensor([False, True, False, True])
            default.reset(mask)
            explicit.reset(mask)
            assert torch.equal(default.stages, explicit.stages)


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_partial_reset_only_clears_selected_return_diagnostics(environment):
    batch = environment(3, seed=22, deadline_mode="terminal")
    for _ in range(2):
        batch.step(batch.teacher())
    names = (
        "episode_return",
        "episode_task_return",
        "discounted_task_return",
        "steps",
        "episode_discount",
    )
    before = {name: getattr(batch, name).clone() for name in names}
    batch.reset(torch.tensor([False, True, False]))
    for name in names:
        assert torch.equal(getattr(batch, name)[[0, 2]], before[name][[0, 2]])
        assert float(getattr(batch, name)[1]) == (1 if name == "episode_discount" else 0)
    _, _, _, _, info = batch.step(batch.teacher())
    assert info["steps"].tolist() == [3, 1, 3]
    torch.testing.assert_close(
        info["discounted_task_return"][1], info["task_reward"][1], rtol=0, atol=0
    )


def test_loading_exact_aero_configs_resets_return_diagnostics():
    batch = BatchedAeroAscent(2, seed=74, deadline_mode="terminal")
    configs = [batch.export_config(index) for index in range(2)]
    batch.step(batch.teacher())
    batch.load_configs(configs)
    assert not bool(batch.steps.any())
    assert not bool(batch.episode_task_return.any())
    assert not bool(batch.discounted_task_return.any())
    assert batch.episode_discount.tolist() == [1, 1]


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"deadline_mode": "other"},
        {"deadline_mode": "truncate", "timeout_penalty": -30},
        {"deadline_mode": "terminal", "timeout_penalty": 1},
        {"deadline_mode": "terminal", "timeout_penalty": float("nan")},
        {"deadline_mode": "terminal", "timeout_penalty": float("inf")},
        {"deadline_mode": "terminal", "timeout_penalty": True},
    ],
)
def test_objective_options_are_validated_before_sampling(environment, kwargs):
    with pytest.raises(ValueError):
        environment(2, **kwargs)
