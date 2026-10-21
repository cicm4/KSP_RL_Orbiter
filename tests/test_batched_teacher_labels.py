"""Batched label reconstruction must retain the serial witness semantics."""

from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from orbiter import distill_ascent as distillation
from orbiter.ascent_planner import SearchBudget, config_fingerprint, plan_ascent
from orbiter.env import EnvConfig, Stage


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def mixed_witnesses():
    two = replace(EnvConfig(), randomization=0, target_altitude=100000,
                  orbit_tolerance=20000, max_time=1600, physics_dt=1)
    three = replace(two, target_altitude=105000, payload_mass=1550, stages=(
        two.stages[0], Stage(600, 600, 120000, 345, 85),
        Stage(400, 5400, 120000, 345, 85),
    ))
    four = replace(two, target_altitude=110000, payload_mass=1600, stages=(
        two.stages[0], Stage(300, 300, 120000, 345, 85),
        Stage(300, 300, 120000, 345, 85), Stage(400, 5400, 120000, 345, 85),
    ))
    configs = [two, three, four]
    report = plan_ascent(configs, budget=SearchBudget(population=8, iterations=2, seed=91))
    assert report["qualified_successes"] == len(configs)
    actions = [torch.tensor(row["actions"]) for row in report["results"]]
    qualifications = [row["qualification"] for row in report["results"]]
    return configs, actions, qualifications, report


def assert_same_labels(actual, expected):
    assert actual.config == expected.config
    assert actual.config_sha256 == expected.config_sha256
    assert actual.vehicle_sha256 == expected.vehicle_sha256
    assert actual.qualification == expected.qualification
    for name in ("observations", "previous_actions", "baseline_actions", "teacher_actions"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)


def test_batched_labels_match_serial_for_mixed_stages_targets_masses_and_lengths(mixed_witnesses):
    configs, actions, qualifications, _ = mixed_witnesses
    assert len(set(map(len, actions))) == 3
    actual = distillation.replay_labels_batch(configs, actions, qualifications)
    for trajectory, config, commands, proof in zip(actual, configs, actions, qualifications):
        assert_same_labels(trajectory, distillation.replay_labels(config, commands, proof))


def test_completed_worlds_do_not_advance_or_extend_their_labels(mixed_witnesses, monkeypatch):
    configs, actions, qualifications, _ = mixed_witnesses
    original_step = distillation.BatchedAeroAscent.step
    completed = None
    frozen = None
    checks = 0

    def checked_step(env, action):
        nonlocal completed, frozen, checks
        result = original_step(env, action)
        if completed is not None and bool(completed.any()):
            torch.testing.assert_close(env.state[completed], frozen[completed], rtol=0, atol=0)
            assert bool((result[-1]["elapsed"][completed] == 0).all())
            checks += 1
        completed = result[2] | result[3] if completed is None else completed | result[2] | result[3]
        frozen = env.state.clone()
        return result

    monkeypatch.setattr(distillation.BatchedAeroAscent, "step", checked_step)
    actual = distillation.replay_labels_batch(configs, actions, qualifications)
    assert checks > 0
    assert [len(row.observations) for row in actual] == list(map(len, actions))
    for row, commands in zip(actual, actions):
        torch.testing.assert_close(row.previous_actions[1:], commands[:-1], rtol=0, atol=0)
        torch.testing.assert_close(row.previous_actions[0], torch.zeros(2), rtol=0, atol=0)


@pytest.mark.parametrize("change", ["extra", "insufficient", "empty", "nonfinite", "unbounded"])
def test_batch_rejects_invalid_or_nonterminal_witness_actions(mixed_witnesses, change):
    configs, original, qualifications, _ = mixed_witnesses
    actions = [row.clone() for row in original]
    if change == "extra":
        actions[0] = torch.cat((actions[0], actions[0][-1:]))
    elif change == "insufficient":
        actions[0] = actions[0][:-1]
    elif change == "empty":
        actions[0] = actions[0][:0]
    elif change == "nonfinite":
        actions[0][0, 0] = float("nan")
    else:
        actions[0][0, 0] = 1.01
    with pytest.raises(ValueError):
        distillation.replay_labels_batch(configs, actions, qualifications)


def test_different_global_configs_are_grouped_without_reordering(mixed_witnesses, monkeypatch):
    original_configs, original_actions, original_qualifications, _ = mixed_witnesses
    # These globals affect environment metadata/rewards without invalidating the
    # already successful physical action sequence. load_configs cannot mix them.
    configs = [original_configs[0], replace(original_configs[1], gamma=.998),
               original_configs[2], replace(original_configs[0], max_time=1800)]
    actions = [original_actions[index] for index in (0, 1, 2, 0)]
    qualifications = [original_qualifications[index] for index in (0, 1, 2, 0)]
    sizes = []
    original_load = distillation.BatchedAeroAscent.load_configs

    def checked_load(env, bank):
        sizes.append(len(bank))
        return original_load(env, bank)

    monkeypatch.setattr(distillation.BatchedAeroAscent, "load_configs", checked_load)
    actual = distillation.replay_labels_batch(configs, actions, qualifications)
    assert sizes == [2, 1, 1]
    for trajectory, config, commands, proof in zip(actual, configs, actions, qualifications):
        assert_same_labels(trajectory, distillation.replay_labels(config, commands, proof))


def test_load_witnesses_requalifies_each_craft_before_one_batched_replay(tmp_path, mixed_witnesses, monkeypatch):
    _, _, qualifications, report = mixed_witnesses
    path = tmp_path / "teachers.json"
    path.write_text(json.dumps(report))
    calls = []
    original_qualify = distillation.qualify_witness
    original_replay = distillation.replay_labels_batch

    def qualify(config, actions, **kwargs):
        calls.append(("scalar", config_fingerprint(config)))
        return original_qualify(config, actions, **kwargs)

    def batch(configs, actions, proofs):
        calls.append(("batch", len(configs)))
        assert all(proof["qualified"] for proof in proofs)
        return original_replay(configs, actions, proofs)

    monkeypatch.setattr(distillation, "qualify_witness", qualify)
    monkeypatch.setattr(distillation, "replay_labels_batch", batch)
    bank = distillation.load_witnesses(path)
    assert [kind for kind, _ in calls] == ["scalar", "scalar", "scalar", "batch"]
    assert len(bank.trajectories) == len(qualifications)

