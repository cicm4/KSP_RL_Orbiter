# KSP RL Orbiter

Accelerated orbital-launch training with a headless Gymnasium simulator and
an external kRPC autopilot for Kerbal Space Program. Train without running the
game, calibrate against actual flight measurements, then validate the learned
controller in KSP. This is a Python tool and library foundation; it is not yet
a packaged in-game autopilot mod.

Two included PPO checkpoints have completed actual staged KSP launches:

| Model | Training transitions | Unseen simulator launches | Actual KSP orbit |
| --- | ---: | ---: | --- |
| [Terrier PPO](models/terrier_ppo.zip) | 131,072 | 100/100 | 72.0 x 81.2 km |
| [Terrier PPO with fitted aerodynamics](models/terrier_aero_ppo.zip) | 114,688 | 99/100 | 71.2 x 85.5 km |

Orbit values are periapsis x apoapsis. Each model has one actual KSP launch on
the same craft, followed by a separate 30–39 second coast check at zero throttle.
This is not a cross-craft reliability estimate or a complete-revolution test.
The [validation manifest](validation/terrier/manifest.json) links model hashes
to the original KSP recordings. Keep each checkpoint with its metadata file.

## Repository

- `orbiter/`: reusable environment, training, calibration and KSP adapters.
- `orbiter/configs/`: measured Terrier craft profiles, included in the package.
- `tests/`: physics, reward, checkpoint, staging and fake-kRPC regression tests.
- `models/`: the two small KSP-validated checkpoints and their metadata.
- `validation/terrier/`: actual KSP telemetry and calibration evidence.
- `legacy/`: the original class project, notebook, paper, figures and historical
  checkpoints, preserved for comparison. Its scripts retain their original
  relative paths; run them from inside `legacy/` with the original dependencies.
- `runs/`: ignored local training output; new runs do not alter legacy data.

## Run

Python 3.12 on macOS was used for validation:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pytest tests -q
.venv/bin/python -m orbiter.train --algo ppo --n-envs 8 --steps 131072 --run-dir runs/ppo
.venv/bin/python -m orbiter.train evaluate --resume runs/ppo/latest.zip --eval-seed 200000 --eval-episodes 100 --run-dir runs/ppo/test
.venv/bin/python -m orbiter.train --resume runs/ppo/latest.zip --n-envs 8 --steps 500000 --run-dir runs/ppo
```

For a minimal library install, use `pip install .`. Optional dependencies are
`.[train]`, `.[ksp]` and `.[dev]`; `pip install -e '.[train,ksp,dev]'` installs a
development checkout without the exact versions in `requirements.txt`.
Installed commands include `orbiter-train`, `orbiter-live`, `orbiter-validate`,
`orbiter-capture`, `orbiter-calibrate`, `orbiter-fit-aero`, `orbiter-make-craft`
and `orbiter-bootstrap`. The `python -m orbiter.<module>` forms also work.

The core environment does not import PyTorch or connect to KSP:

```python
from orbiter import EnvConfig, HeadlessOrbiterEnv

env = HeadlessOrbiterEnv(EnvConfig())
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step([1.0, 1.0])
env.close()
```

Default PPO starts from 30 scripted demonstration flights, behavior cloning,
and critic fitting to complete discounted returns. GAE lambda is 1, gamma is
0.999 per two-second control decision, initial action standard deviation is
`exp(-3)`, and learning rate is `3e-5`. These settings are explicit CLI options.
Use `--bc-episodes 0` for an experiment without demonstrations. `--algo sac`
and `--algo tqc` provide off-policy comparisons; they require their own tuning.
Training is CPU based with one PyTorch thread per process.

Each run contains configuration, JSONL episode/evaluation/training metrics,
random/scripted baselines, `latest.zip`, `best.zip`, and checkpoint metadata.
Selection uses fixed evaluation seeds; evaluate the final checkpoint on a
different seed range before reporting results. SIGINT/SIGTERM checkpoint the
run. Off-policy continuation can save/reload replay with `--save-replay` and
`--load-replay`; resume rejects a changed environment/reward contract. A resume
restores policy/optimizer state, not a bit-for-bit simulation/RNG continuation.

## Task and simulator

Actions are throttle and an absolute autopilot pitch setpoint, with heading
fixed east. They are not the raw pitch/yaw torque controls used in the old
project. Old demonstrations/checkpoints are incompatible with the new contract.
Observations include inertial vertical/east speed, orbital energy/apses, fuel,
mass, stage, attitude, throttle, thrust-to-weight, time, and dynamic pressure.

Success is a bound orbit with periapsis at least 71 km and apoapsis at most
90 km for the default 80 km target. Reward is +100 for success, -30 for failure,
a small time cost, and `gamma * Phi(next) - Phi(current)` shaping using orbital
energy and angular momentum. Genuine terminals use zero next potential;
time truncations preserve the next potential and bootstrap. An altitude hop
alone cannot qualify as orbit. Potential shaping does not guarantee successful
exploration or prevent function-approximation failure.

The simulator includes inverse-square gravity, planet/atmosphere rotation,
pressure-dependent thrust/Isp, propellant mass flow, sequential automatic
staging, drag, optional angle-dependent drag/lift, attitude slew/lag, impact/escape detection and substepped
integration. Tests cover orbital conservation, powered timestep convergence,
mass/staging, reward terminal semantics and held-out scripted launches.

The default vehicle is explicitly **generic**, with a 1,500 kg payload and two
stages: 3,500/22,000 kg dry/propellant at 650 kN vacuum thrust; then 1,000/6,000 kg
at 120 kN. It is not the original class-project rocket. Default atmosphere and
drag are approximations until calibrated. This is a planar surrogate, not exact
KSP/Unity physics: it omits drag cubes/occlusion, full aerodynamic torque, joints,
heating, terrain, crossfeed and six-degree-of-freedom attitude dynamics.

## KSP capture and validation

Start the installed KSP with kRPC enabled and enter a flight/save. Capture is
read-only and does not quicksave, quickload, stage, or change controls:

```sh
.venv/bin/python -m orbiter.capture --output runs/ksp_capture.json --samples 10
.venv/bin/python -m orbiter.calibrate --capture runs/ksp_capture.json --output runs/kerbin.json
.venv/bin/python -m orbiter.live --config runs/kerbin.json
.venv/bin/python -m orbiter.live --help
```

Calibration imports body constants and atmospheric pressure/density in SI units.
Its provenance file distinguishes imported data from retained generic craft
parameters. It does not automatically infer a rocket from part masses: verify
stage dry/propellant masses, engines and crossfeed, then supply a matching
`EnvConfig` JSON via `--config`. Optional `--fit-drag` requires coherent in-flight
measurements. Held-out KSP flights and measured trajectory error are required to
establish transfer; simulator success is not evidence of KSP success.

`orbiter.live` defaults to inspection. An explicit `--fly` executes either a
model checkpoint or `--scripted` guidance, with a checked activation-stage plan
and tank decouple-stage mapping. It tracks the control part across separations,
uses bounded simulation-time waits, records commanded actions/telemetry, and
cuts throttle/disengages autopilot in cleanup. The initial adapter supports
sequential liquid stages and rejects solid/airbreathing engines and ambiguous
parallel ignition groups. Inspect stage numbers before supplying a flight plan.
It never implicitly overwrites the user's quicksave.
Learned-policy flight also rejects wet/stage propellant masses more than 10%
from its training configuration. This catches obvious craft mismatches, not
unmodeled engine/aerodynamic differences.

`orbiter.make_craft` derives an 18-part conventional liquid rocket from the
installed stock `Orbiter One.craft`. With `--terrier-template` pointing to stock
`Science Jr.craft`, it creates `OrbiterRL Terrier`, replacing the heavy upper
Swivel with a Terrier and matching both stack attachment joints. It refuses to
overwrite files. Use the dedicated `OrbiterRL` save; the original CS5100 save
on this Mac contains an Aeris plane and remains unchanged.

The tested craft profile is `orbiter/configs/orbiter_rl_terrier.json`: 12,060 kg
wet mass, 6,000/2,000 kg stage propellant, a 215 kN Swivel and a 60 kN Terrier.
Its body constants, atmosphere, masses, engines and stage mapping were checked
through live kRPC. Activation stages are `3,2`; tank decouple stages are `2,1`.
Generate the matching craft from your own KSP installation, after creating a
dedicated `OrbiterRL` game save:

```sh
KSP_ROOT="/path/to/Kerbal Space Program"
.venv/bin/python -m orbiter.make_craft --source "$KSP_ROOT/Ships/VAB/Orbiter One.craft" --terrier-template "$KSP_ROOT/Ships/VAB/Science Jr.craft" --output "$KSP_ROOT/saves/OrbiterRL/Ships/VAB/OrbiterRL Terrier.craft"
```

Load the generated craft in the VAB, launch it onto the pad, enable the kRPC
server, and inspect with `python -m orbiter.live` before flying. The generic
checkpoint does not match this craft. To train a separate Terrier policy:

```sh
.venv/bin/python -m orbiter.train --config orbiter/configs/orbiter_rl_terrier.json --n-envs 8 --steps 131072 --run-dir runs/ppo_terrier
.venv/bin/python -m orbiter.live --fly --model runs/ppo_terrier/best.zip --stage-plan 3,2 --fuel-stages 2,1
```

Native save selection was unresponsive on this Mac. The optional
`orbiter/bootstrap.py` compiles an addon using the installed Mono `mcs` and KSP
assemblies. Install with `--ksp-root <KSP directory> --install`, then launch
KSP with `-orbiter-bootstrap` to enter only `OrbiterRL`. Without that flag the
addon is inert. Remove `GameData/OrbiterRL/OrbiterBootstrap.dll` to uninstall.
This uses the same GamePersistence APIs as
[AutoLoadGame](https://github.com/allista/AutoLoadGame/blob/master/AutoLoadGame.cs).

Actual KSP testing exposed and fixed two adapter bugs: re-engaging autopilot
each decision reset its PID state; reading attitude from the planet's frame
gave about 15 degrees pitch for a vertically parked rocket. Live attitude now
uses the surface frame and autopilot retains its controller between decisions.
The scripted baseline completed a KSP orbit at 71.4 x 80.9 km in 251 seconds;
an independent read 36 seconds after cutoff confirmed throttle zero and a
71.5 x 80.9 km orbit. Its recording is
`runs/live/terrier_scripted_01.jsonl`. This baseline does not establish neural
policy transfer; learned-policy KSP attempts are recorded separately.

The craft-specific PPO checkpoint in `models/terrier_ppo.zip` completed
an actual KSP launch with automatic staging in 273 seconds, ending at a
72.0 x 81.2 km orbit. A separate coast read confirmed throttle zero and the
same safe orbit. `validation/terrier/terrier_policy_01.jsonl` contains commands
and telemetry; the validation manifest identifies the exact checkpoint hash. Keep
each model `.zip` with its sibling `.metadata.json` when moving it.

Actual ascent forces revealed that constant drag alone missed fin lift. Fit
the optional planar drag/lift law using a captured flight:

```sh
.venv/bin/python -m orbiter.fit_aero --capture validation/terrier/ksp_terrier_ascent_capture.json --base-config orbiter/configs/orbiter_rl_terrier.json --output runs/terrier_aero_fit.json
.venv/bin/python -m orbiter.train --config runs/terrier_aero_fit.json --n-envs 8 --steps 131072 --run-dir runs/ppo_terrier_aero
```

The fitter reserves every fourth coherent first-stage sample and writes source
hashes, fitting/held-out indices, force errors and scope into provenance JSON.
It reports planar lift error, omitted orthogonal lift, and full lift-vector
error separately so a good planar fit cannot hide strong out-of-plane forces.
The fitted profile is also provided as
`orbiter/configs/orbiter_rl_terrier_aero.json`. This is a local approximation
within the measured flight regime. It applies coefficients globally; later-stage
geometry, high-angle/Mach extrapolation and 3D structural dynamics remain outside
its validation. A fit on one flight cannot substitute for testing the resulting
policy in KSP.

The resulting fitted-aero PPO checkpoint is
`models/terrier_aero_ppo.zip` (114,688 training transitions; selected
using fixed simulation validation seeds). It passed 99/100 unseen simulator
episodes and then completed an actual staged KSP launch in 303.1 seconds,
reaching 71.16 x 85.48 km. This flight used the neural policy directly for
throttle/pitch, with kRPC autopilot tracking attitude and explicit stage control.
`validation/terrier/terrier_aero_policy_01.jsonl` records its checkpoint SHA256, actions,
staging, telemetry and successful termination. A separate read-only, 30-second
KSP coast check confirmed 71.19 x 85.49 km, zero throttle and an orbiting vessel:
`validation/terrier/terrier_aero_policy_01_coast.json`. On the validation Mac,
the final orbit is saved as `OrbiterRL-validated-orbit` in the dedicated save.
The temporary bootstrap DLL was removed after testing; its source/build command
remains available.

An independent ascent capture also checked the frozen aerodynamic coefficients,
without refitting. Over the first 87.4 seconds of commanded flight replay,
altitude RMSE fell from 1,077 m to 45 m; drag-force RMSE fell from 12,183 N to
1,304 N. Full scope, source hashes and errors are in
`validation/terrier/aerodynamics_independent.json`. These measurements establish
local first-stage agreement, not exact KSP physics. The live tests establish
successful transfer for this craft, with one launch per neural checkpoint;
they do not establish a cross-craft success rate or a complete-orbit coast test.

To repeat a flight and require an actual KSP coast check, explicitly reload the
named dedicated pad snapshot (this replaces the current active flight). The
snapshot is local to the validation Mac. On another installation, prepare the
matching craft in a dedicated save and use `orbiter.live --fly` with
`--save-name OrbiterRL-pad-terrier` to create your own preflight snapshot:

```sh
.venv/bin/python -m orbiter.validate --model models/terrier_aero_ppo.zip --pad-save OrbiterRL-pad-terrier --stage-plan 3,2 --fuel-stages 2,1 --coast-seconds 30 --output-dir runs/ksp_repeat_01
```

The validator has no headless fallback and exits nonzero on flight/coast
failure. `--verify-current` instead reads the current orbit without loading,
staging or flying; that mode alone cannot attribute an orbit to a controller.
Use a new output directory for each validation. Keep exported model metadata
with its checkpoint. New training output stays in ignored `runs/`; the curated
models and KSP evidence above are tracked separately. KSP, stock craft files and
third-party mods are not distributed with this Python package.

## Research informing this implementation

- [MIT Lincoln Laboratory KSPDG](https://github.com/mit-ll/spacegym-kspdg): robust
  KSP/Gym integration, with KSP intended for evaluation rather than accelerated
  parallel training. Its scenarios are orbital pursuit/evasion, not ascent.
- [KSP-RL-Agents](https://github.com/Weirdnemo/KSP-RL-Agents): related PPO launch,
  hover and landing examples; full orbital insertion is not demonstrated there.
- [Space-Gym](https://github.com/MIMUW-RL/space-gym): accelerated planar orbital
  environments and examples of task/reward sensitivity across algorithms.
- [Potential shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),
  [episodic terminal handling](https://aamas.csc.liv.ac.uk/Proceedings/aamas2017/pdfs/p565.pdf),
  and [Gymnasium truncation guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
  inform the reward/bootstrapping contract.
- [TQC](https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html) supplies a
  maintained distributional SAC comparison. [CrossQ](https://github.com/adityab/CrossQ),
  [SimbaV2](https://github.com/DAVIAN-Robotics/SimbaV2), and
  [XQC](https://github.com/danielpalenicek/xqc) are stronger modern research
  candidates for subsequent controlled ablations; their benchmark claims do not
  establish performance on KSP launch.
- [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) and
  [TD-MPC2](https://arxiv.org/abs/2310.16828) offer learned models/planning but add
  complexity before this small physical simulator becomes a sampling bottleneck.
  [Dynamics randomization](https://arxiv.org/abs/1710.06537) motivates varying
  physical parameters; uncertainty should ultimately come from KSP measurements.
