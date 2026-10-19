# KSP RL Orbiter

Accelerated orbital-launch training with a headless Gymnasium simulator and
an external kRPC autopilot for Kerbal Space Program. Train without running the
game, calibrate against actual flight measurements, then validate the learned
controller in KSP. A stock-toolbar KSP addon provides the in-game controls and
starts the local Python flight engine. Training and policy execution stay in
Python, connected to the game through kRPC.

Three included ascent checkpoints have completed actual staged KSP launches:

| Model | Training transitions | Unseen simulator launches | Actual KSP orbit |
| --- | ---: | ---: | --- |
| [Terrier PPO](models/terrier_ppo.zip) | 131,072 | 100/100 | 72.0 x 81.2 km |
| [Terrier PPO with fitted aerodynamics](models/terrier_aero_ppo.zip) | 114,688 | 99/100 | 71.2 x 85.5 km |
| [Transfer-craft PPO](models/transfer_ascent_ppo.zip) | 131,072 | 64/64 | 72.0 x 146.0 km |

Orbit values are periapsis x apoapsis. The first two models each have one actual
launch on the 18-part Terrier, followed by a separate 30–39 second coast check at
zero throttle. The third uses the larger 20-part transfer craft and a broader
safe parking-orbit envelope. These counts are not cross-craft reliability estimates.
The [validation manifest](validation/terrier/manifest.json) links model hashes
to the original Terrier recordings. Selected-destination validation is described
below. Keep each checkpoint with its metadata file.

## Repository

- `orbiter/`: reusable environment, training, calibration and KSP adapters.
- `orbiter/configs/`: measured Terrier craft profiles, included in the package.
- `tests/`: physics, reward, checkpoint, staging and fake-kRPC regression tests.
- `models/`: portable controller checkpoints and their task/craft metadata.
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
`orbiter-capture`, `orbiter-calibrate`, `orbiter-fit-aero`, `orbiter-make-craft`,
`orbiter-train-capture`, `orbiter-mission`, `orbiter-control`, `orbiter-mod` and `orbiter-bootstrap`.
The `python -m orbiter.<module>` forms also work.

To fly through a control panel, place the **OrbiterRL Transfer** rocket on Kerbin's
launchpad and start the kRPC server in KSP. Open
[Start Autopilot.command](<Start Autopilot.command>) on macOS, or run:

```sh
.venv/bin/python -m orbiter.control --project-root . --launch-model models/transfer_ascent_ppo.zip --capture-model models/moon_capture.zip
```

Choose **Mun** or **Minmus**, set a **40–60 km** destination orbit, and press
**Run**. The active ship is checked before launch. The controller handles staging,
parking orbit, transfer, capture, and a full-period orbit check. **Stop** releases
the controls back to KSP. Flight progress and results appear in the panel;
recordings are saved in `runs/control/`. The panel supports the trained 20-part
transfer craft and these two destinations. It does not support arbitrary ships
or interplanetary launches.

Actual KSP Run-button checks completed the entire flight from the existing pad
ship: a **50 km Mun goal reached 49.43 x 54.86 km**, and a **60 km Minmus goal
reached 59.65 x 64.96 km**. Each then passed one full independent orbital period
at zero throttle (3,117 and 6,395 seconds). Staging, ascent, transfer and capture
ran without manual flight inputs. Refreshing the panel during flight preserves
the locked destination and altitude; the flight continues independently of the page.

The first Minmus 60 km attempt stopped safely after ascent because a node-vector
check compared different-time inertial readings. The corrected check verifies
the encoded burn in the same node frame, retaining strict axis/sign checks and
KSP encounter verification. A fresh launchpad Run then passed. The
[mission manifest](validation/missions/manifest.json) preserves both successful
button flights, that failed attempt, the diagnostic, requests and final results.
These individual demonstrations do not establish a repeated-flight success rate.

### KSP toolbar addon

With the Python environment above, the Mono C# compiler (`mcs`), and kRPC
installed in KSP, build and install against your own game assemblies:

```sh
.venv/bin/python -m orbiter.mod --ksp-root "$KSP_ROOT" --project-root "$PWD" --python "$PWD/.venv/bin/python" --install
```

Restart KSP, enter flight, and click the **Orbiter** icon on the stock sidebar.
The game window provides destination, altitude, **Run**, **Stop**, and flight
status. It starts the configured local engine automatically; no browser window
is required. Keep the kRPC server running. The same matched transfer rocket,
Mun/Minmus destinations, and 40–60 km goal limits apply.

A separate native-toolbar Run completed the full pad-to-Mun flight at the
selected 50 km goal, reaching 49.35 x 54.91 km and then coasting unpowered for
3,116.68 seconds. The first client lost its displayed job identity during that
flight; the engine continued and passed every orbit/cleanup check. The corrected
client uses KSP's managed JSON decoder with explicit identity/proof checks. A
cold reload read the completed flight proof successfully. The
[native flight recording](validation/missions/mun_native_launch.json) preserves
the engine result separately from frontend validation.

The installer writes only `GameData/OrbiterAutopilot`: the addon DLL, runtime
settings, and an install receipt. The engine's output goes to `backend.log` there;
flight recordings remain under the configured project's `runs/control/`.
Absolute interpreter/project paths are stored in `settings.cfg`; rerun the
installer with `--replace` after moving the environment or updating the addon.
Replacement requires a matching install receipt and preserves backups with
extensions KSP will not load as plugins. Game assemblies are never redistributed.

Closing the window leaves the flight running. Stop waits for the engine's
control-release result; a lost connection is not reported as a successful stop.
The backend verifies its identity and flight job before accepting a native Stop,
and an OS ownership lock prevents overlapping Orbiter mission workers after a
backend restart. The optional validation-save bootstrap remains a separate addon
with its own explicit startup flag.

### Release package and license

Package the current compiled addon with the Python source and the two validated
live policy pairs:

```sh
.venv/bin/python -m orbiter.release_mod --assembly runs/mod/OrbiterAutopilot.dll --output runs/release/OrbiterAutopilot-0.1.0.zip
```

The ZIP contains `GameData/OrbiterAutopilot`, `OrbiterEngine`, `install.py`,
installation instructions, and a manifest of file hashes. Extract the complete
archive into a permanent directory, then run `python3 install.py --ksp-root
"<your KSP directory>"`. The installer prepares a Python environment, downloads
its dependencies, and configures the prebuilt DLL with your local paths. Python,
kRPC, KSP and Unity binaries are not bundled; copying GameData alone does not
configure the Python engine. The packager checks the live models against the
validation manifest and excludes the experimental capture policy.

Normal installation derives the exact 20-part `OrbiterRL Transfer` rocket from
your installed stock `Orbiter One.craft` and `Science Jr.craft`, checking its
14,310 kg wet mass and staging profile. It creates a new file at the extracted
release's `Ships/OrbiterRL Transfer.craft`; `--craft-output` selects another new
path. Copy that file into your save's `Ships/VAB/`, load it in the VAB, assign
crew, and launch onto the pad. Existing craft and stock templates are never
overwritten. Use `--skip-craft` if you already have the supported rocket or the
stock templates are unavailable. Stock-derived craft are not bundled in the ZIP.

The new addon, Python engine and newly trained checkpoints use the
[MIT license](LICENSE), copyright 2026 Camilo Correa Llano. Historical material
under `legacy/`, including the class paper, retains its existing terms.
Third-party software and game assets retain their respective licenses.

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
This craft-specific trainer uses one PyTorch CPU thread per process.

Each run contains configuration, JSONL episode/evaluation/training metrics,
random/scripted baselines, `latest.zip`, `best.zip`, and checkpoint metadata.
Selection uses fixed evaluation seeds; evaluate the final checkpoint on a
different seed range before reporting results. SIGINT/SIGTERM checkpoint the
run. Off-policy continuation can save/reload replay with `--save-replay` and
`--load-replay`; resume rejects a changed environment/reward contract. A resume
restores policy/optimizer state, not a bit-for-bit simulation/RNG continuation.

### Experimental spacecraft generalization

`orbiter.train_generalized` trains a separate policy conditioned on
ordered stage masses, thrust, Isp, available delta-v, burn duration, drag and
attitude response. Float64 RK4 physics, observations, demonstrations and PPO
updates stay on the selected GPU. The original v3 experiment has 54 inputs and
constant drag; `--physics aero` selects v4 with 56 inputs, angle-dependent drag
and signed lift. V4 trains all supported two-, three- and four-stage layouts.
Two jobs can use both GPUs:

```sh
.venv/bin/python -m orbiter.train_generalized --run-dir runs/generalized-stock --device cuda:0 --curriculum stock --seed 404 --worlds 8192 --label-steps 800 --bc-updates 10000 --dagger-rounds 10 --ppo-updates 500 --batch-size 16384 --eval-worlds 512
.venv/bin/python -m orbiter.train_generalized --run-dir runs/generalized-efficient --device cuda:1 --curriculum efficient --seed 303 --worlds 8192 --label-steps 800 --bc-updates 10000 --dagger-rounds 10 --ppo-updates 500 --batch-size 16384 --eval-worlds 512
```

Run these in separate terminals or process sessions. The v3 constant-drag
collection measured about 340,000 control transitions per second per RTX 4090
at this batch size. Checkpoints are saved during training and on SIGINT/SIGTERM.
`--resume` loads matching weights with fresh optimizers and simulator worlds;
use a new, empty output directory. `--ppo-only` skips imitation when continuing
initialized weights. Evaluation rejects mismatched physics or curriculum.

To initialize the aero experiment from a stock v3 checkpoint, preserve its
existing input weights and add two zero-weight aerodynamic inputs:

```sh
.venv/bin/python -m orbiter.train_generalized --run-dir runs/generalized-aero-stock --device cuda:0 --physics aero --curriculum stock --transfer-from runs/generalized-stock/best.pt --ppo-only --seed 505 --worlds 8192 --label-steps 800 --value-updates 5000 --ppo-updates 500 --batch-size 16384 --eval-worlds 512 --validation-interval 20 --target-kl 0.02
```

This starts a new task with recorded parent ancestry, not a restored training
session. PPO records approximate KL, clipping, action standard deviation and
optimization epochs; the KL limit is checked before each optimizer step and
can stop the remaining minibatches.
Validation frequency is configurable so evaluation need not dominate training.

To compare deadline objectives, run the finite three-arm, three-seed suite on
two available GPUs:

```sh
.venv/bin/python -m orbiter.ablation_suite --run-root runs/deadline-ablation --actor-from runs/generalized-stock/best.pt --parent-sha256 YOUR_VERIFIED_PARENT_SHA256 --gpus 0 1
```

Replace the hash with the verified parent checkpoint's SHA256. The suite compares
the original time-limit bootstrap, a terminal deadline with zero timeout reward,
and a terminal deadline with a -30 failure reward. All retain the existing time
cost. Every arm copies the same actor into a fresh, identically seeded critic;
validation and comparison craft seeds are separate from training seeds. Defaults
run 100 PPO updates for each of nine jobs, with one child per GPU. Parent/source
hashes are checked before each launch. A failed child stops the queue; SIGTERM
allows checkpointing before bounded termination. Use a new empty output root.
`--plan-only` writes the commands without starting training.

Finite deadlines use explicit v5 objective contracts. Original v3/v4 defaults
remain unchanged. `--actor-from` is an intentional actor-only initialization,
whereas `--warm-start` requires a matching complete contract. PPO diagnostics
include separate actor/critic gradient norms, value explained variance, TD
errors, action saturation and cumulative optimizer steps. The suite's paired
comparison report is research evidence, not release qualification or permission
to install experimental weights.

The independent shooting teacher searches bounded ascent, coast and
circularization feedback parameters using the complete remaining stage stack:

```sh
.venv/bin/python -m orbiter.ascent_planner --help
```

Its benchmark compares the planner and existing heuristic on the same frozen
craft. Successful witnesses are replayed in the scalar simulator and coasted
unpowered for a full orbital period. A failed search leaves reachability unknown.
The planner is experimental and does not control the game.

To compare feedforward and recurrent residual students, generate disjoint teacher
banks and run the separate imitation experiment:

```sh
.venv/bin/python -m orbiter.ascent_planner --count 64 --seed 920 --population 16 --iterations 3 --max-dynamic-pressure 60000 --output runs/teacher-train.json
.venv/bin/python -m orbiter.ascent_planner --count 16 --seed 921 --population 16 --iterations 3 --max-dynamic-pressure 60000 --output runs/teacher-validation.json
.venv/bin/python -m orbiter.distill_ascent --train-report runs/teacher-train.json --validation-report runs/teacher-validation.json --run-dir runs/student-comparison --epochs 40
```

Only freshly scalar-qualified teacher trajectories become labels. Physical craft
identities must differ between banks. Both students augment the same fixed
stage-aware feedback controller, without optimized per-craft parameters at
inference. The GRU trains on ordered sequences and resets its state per craft.
Validation action error selects the epoch; closed-loop validation includes every
validation craft and requires scalar replay, pressure compliance and a full
unpowered orbit. Lower imitation error alone does not establish flight
reliability. Previous actions are teacher-forced during training, so exposure
bias remains; final unseen tests and actual KSP flights are still required.
These checkpoints have a separate experimental contract and no live adapter.

This is an ascent curriculum for screened sequential liquid rockets, with
fresh craft, drag, actuator and captured-craft stress tests. Four-stage tests
are stage-count extrapolation for v3 and fresh-craft tests for v4. The captured
Transfer stress retains measured aerodynamic coefficients and atmosphere in
the scalar simulator. V4 represents angle-dependent effective areas but omits
KSP's part-, Mach- and stage-dependent aerodynamics. SRBs, crossfeed, asymmetric
thrust and aircraft are unsupported. Capture and whole-mission spacecraft
generalization remain unqualified; experimental checkpoints need actual KSP
validation before they can replace the released policies.

The earlier pilot selected imitation actors despite completing 52.4 million PPO
transitions per run: unseen nominal launches passed 203/256
and 101/256 for the efficient and stock-like curricula. Captured Transfer stress
passed 0/8 and 1/8. Those pilot artifacts remain outside the live model defaults;
the corrected v3 experiments need their own evaluation and multiple distinct
actual KSP craft before promotion. Simulator success alone cannot qualify a new
spacecraft for the native Run button.

## Selected-destination missions

The mission extension separates ascent, encounter navigation, and local capture.
`SystemCatalog` contains the 17 stock bodies, their hierarchy, measured constants,
and three-dimensional ephemerides captured from the installed KSP. The pure
planner supports a spacecraft already orbiting the destination's parent:
Kerbin to Mun/Minmus, Jool to its moons, or heliocentric orbit to a planet.
Sibling-planet window searches are **estimates**, not an executable Kerbin-to-Duna
mission. Catalog coverage alone does not establish complete mission coverage.

`CaptureEnv` is a separate 3D arrival task with 27 body-normalized observations
and three acceleration components in the local radial/tangential/normal frame.
It starts near the destination on an incoming trajectory; it does not learn
launch, a transfer window, or SOI acquisition. Goals specify an orbital altitude
and plane. Success requires a bound, safe orbit with both apses inside the target
band and the requested plane, held for two decisions. The reward uses terminal
success/failure, time/fuel costs, and potential shaping. Gravity-scaled action
authority keeps low-gravity bodies from requiring extremely small network
outputs. Finite thrust, fuel use, pointing slew, and substepped gravity are
modeled; this is still a surrogate.

Training uses PPO with behavior cloning and optional
[DAgger](https://proceedings.mlr.press/v15/ross11a) recovery demonstrations.
The latter labels states visited by the learned policy, addressing the drift
seen with demonstrations alone. Checkpoints record the task version, catalog,
body sets and environment contract; incompatible ascent/capture checkpoints
are rejected. Held-out bodies never select the best checkpoint. Fixed-seed
validation, fresh-seed evaluation, and arrival/craft stress tests are separate.

```sh
# New experiment at the expected remaining mass and destination arrival speeds.
# Minmus is trained here, not a held-out body.
orbiter-train-capture --train-bodies Mun,Minmus --heldout-bodies Ike --mass 2500 --dry-mass 1980 --randomization .1 --arrival-vinf-by-body Mun=0.7,Minmus=1.9 --bc-episodes 128 --dagger-rounds 8 --steps 131072 --run-dir runs/capture_moons
orbiter-train-capture evaluate --resume runs/capture_moons/best.zip --eval-seed 400000 --eval-episodes 32 --stress --run-dir runs/capture_moons/test
```

Arrival speed and remaining mass must cover the actual transfer. Passing those
nominal simulator cases does not establish reliability at a faster Minmus
arrival or after a more expensive departure burn. Use `--arrival-vinf-fraction`,
`--arrival-vinf-by-body`, `--mass`, `--dry-mass`, and `--randomization` to define a feasible training regime;
check the rocket-equation budget before training or flying.

`orbiter-mission` defaults to a read-only plan and fuel-budget check. `--fly`
explicitly runs the actual game. Before departure it requires KSP's own patched
conics to predict the requested SOI and safe periapsis, refines the node, executes
a bounded finite burn, and checks the actual SOI transition. Capture uses either
the selected neural policy or an explicitly named scripted baseline. Success
also requires an independent full orbital period in KSP at zero throttle, with
four on-rails coast probes. JSON/JSONL evidence records model hashes, commands,
handoffs, cleanup, and observed orbits; a node prediction alone is not success.
If a direct route crosses another moon first, the planner checks up to eight
parking-orbit departure windows. Rejected candidates are removed without a
departure burn; an exhausted search stops with controls released.

```sh
# Active vessel must already be in a safe Kerbin parking orbit.
orbiter-mission --target Mun --target-altitude 50000 --output-dir runs/mun_plan
orbiter-mission --fly --target Mun --target-altitude 50000 --capture-model models/moon_capture.zip --output-dir runs/mun_live

# Fresh launch through independently verified target orbit, in one connection.
orbiter-mission --fly --target Minmus --target-altitude 50000 --launch-model models/transfer_ascent_ppo.zip --launch-craft 'OrbiterRL Transfer' --stage-plan 3,2 --fuel-stages 2,1 --capture-model models/moon_capture.zip --output-dir runs/minmus_live

# Use the compatible ship already on the launchpad, as the Run button does.
orbiter-mission --fly --target Mun --target-altitude 50000 --launch-model models/transfer_ascent_ppo.zip --launch-active --stage-plan 3,2 --fuel-stages 2,1 --capture-model models/moon_capture.zip --output-dir runs/mun_active_pad
```

Use a craft-matched ascent model and a capture policy validated for the planned
arrival. The 18-part Terrier has limited transfer reserve. `orbiter-make-craft
--transfer` adds two upper-stage FL-T200 tanks to the generated Terrier profile:
20 parts, 14,310 kg wet, 6,000/4,000 kg propellant, and the same `3,2` activation /
`2,1` fuel-stage mapping. Its ascent checkpoint must use the larger craft profile.
The launcher preserves existing vessels (`recover=False`), rejects existing
maneuver nodes, and does not implicitly overwrite quicksave or persistent saves.
An optional `--pad-save-name` requires `--ksp-root` and a new dedicated save name.
To repeat a complete flight from an existing crewed pad state, replace
`--launch-craft` with `--launch-pad-save NAME --ksp-root "$KSP_ROOT"`. This accepts
only a named pre-launch snapshot in the dedicated `OrbiterRL` save and verifies
the loaded craft's staging and wet mass. It cannot accompany `--pad-save-name`.
The active KSP game must be `OrbiterRL` when using this snapshot route.

After generating the Terrier as described below, create the transfer variant
and train against its packaged profile:

```sh
orbiter-make-craft --transfer --source "$KSP_ROOT/saves/OrbiterRL/Ships/VAB/OrbiterRL Terrier.craft" --output "$KSP_ROOT/saves/OrbiterRL/Ships/VAB/OrbiterRL Transfer.craft"
orbiter-train --config orbiter/configs/orbiter_rl_transfer.json --n-envs 8 --steps 131072 --run-dir runs/transfer_ascent
```

The included `moon_capture.zip` was initialized from a 131,072-step PPO actor,
then adapted with DAgger for the transfer craft's remaining mass and arrival
speeds. A further 131,072-step PPO experiment completed, but train-only checkpoint
selection retained the actor before that new PPO phase (`num_timesteps=0`).
Its fresh simulator evaluation passed 32/32 each for trained Mun and Minmus and
held-out Ike. Eighteen arrival/mass settings each passed the same eight stress
seeds; those are 18 settings x 8 seeds, not 144 independent random missions.
These tests cover local capture, not launch or transfer reliability.

The separate `stock_capture_experimental.zip` was trained on all 16 orbitable
stock bodies. Fresh nominal capture tests passed 511/512: 32/32 on fifteen
bodies and 31/32 on Jool (one timeout). Its generic vehicle is 4,000 kg wet /
1,000 kg dry, 60 kN and 345 s ISP, with nominal arrival excess speed 0.4 times
local circular speed. This differs from the transfer craft and moon-arrival
policy. All 16 bodies are trained here; this is not zero-shot body evidence.
The selected actor uses BC and DAgger with no PPO ancestry. The additional
65,536-step PPO experiment regressed on fixed validation, so its updates were
not selected. Separate stresses found Gilly timeouts at higher goals and Jool
time/fuel failures. The checkpoint remains experimental and has no live-game
validation. Full nominal/stress results and sampling details are linked from
the [training evidence](validation/missions/training.json).

In actual KSP, this policy completed the Kerbin-parking-orbit to Mun mission at
49.3 x 55.0 km, followed by a full 3,117-second revolution with throttle zero and
autopilot disengaged. The launch was recorded separately. The Minmus flight ran
from a saved crewed pad through learned ascent and staging, transfer, and learned
capture on one connection, reaching 49.57 x 54.88 km. It then passed a full
5,621-second revolution at zero throttle. Four candidate departure windows were
rejected before KSP verified a direct encounter on the fifth. These are individual
actual demonstrations, not an estimated repeated-flight success rate. A scripted
arrival baseline reached 47.9 x 55.0 km and passed its own full-revolution coast check;
its controls and result are explicitly distinguished from the learned policy.
The [mission manifest](validation/missions/manifest.json) links controller hashes,
compressed game logs, training ancestry and independent evaluations. Recompute
the local vacuum diagnostic without opening KSP:

```sh
python validation/missions/check_vacuum.py validation/missions/mun_scripted_arrival.jsonl.gz
```

That single-flight check compares logged mass flow, inferred powered impulse,
and four short zero-thrust coasts. The fitted command/mass time offset conflates
actuator and sequential RPC timing; it is not an independent latency measurement
or an exact-Unity-physics claim.

The architecture follows goal-conditioned control
([UVFA](https://proceedings.mlr.press/v37/schaul15.html)) while retaining explicit
orbital mechanics for long coasts. Similar practical navigation work includes
[MechJeb](https://github.com/MuMech/MechJeb2) and
[Kerbal Transfer Illustrator](https://github.com/theastrogoth/Kerbal-Transfer-Illustrator).
More recent [contrastive goal RL](https://arxiv.org/abs/2206.07568) and
[hindsight replay](https://proceedings.neurips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html)
are research alternatives; they are not implemented here or evidence that
arbitrary destinations are reliable. The current deployment is an external
kRPC autopilot, with in-game mod packaging still separate work.

For larger experiments, the research shortlist also includes
[JaxGCRL (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/2858e880333b3cd64f8192f13ddcca2f-Paper-Conference.pdf),
which batches goal-conditioned simulation and replay on a GPU,
[equivariant contrastive RL (2025)](https://arxiv.org/abs/2507.16139), which studies
rotation symmetry in manipulation, and
[TD-MPC2](https://arxiv.org/abs/2310.16828), which combines a learned world model
with local planning. These results motivate future scaling/representation
experiments; they do not validate spacecraft transfer. Here vacuum dynamics
are already inexpensive and known, so the first implementation uses analytic
navigation and a small learned capture policy, with KSP measurements testing
the remaining simulator mismatch.

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
- [MetaMorph](https://arxiv.org/abs/2203.11931) motivates conditioning on vehicle
  structure; this project's masked stage descriptors use a small MLP.
  [RMA](https://arxiv.org/abs/2107.04034)
  is a candidate for future adaptation to unknown dynamics. [Isaac Gym](https://arxiv.org/abs/2108.10470)
  motivates keeping simulation and learning data on the GPU. Our physics is a
  PyTorch RK4 implementation with the calibration limits described above.
