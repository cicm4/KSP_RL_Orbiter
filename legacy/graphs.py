
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

#Configuration
DATA_DIR = Path("data/logs")
OUT_DIR = Path("figures")
MAX_EPISODE = 300
SMOOTH_WINDOW = 20   # rolling-average window for noisy curves

OUT_DIR.mkdir(exist_ok=True)

# Load data
ep_df = pd.read_csv(DATA_DIR / "training_episode_metrics.csv")
st_df = pd.read_csv(DATA_DIR / "training_step_metrics.csv")

# Filter to episodes 0-300
ep_df = ep_df[ep_df["episode"] <= MAX_EPISODE].copy()
st_df = st_df[st_df["episode"] <= MAX_EPISODE].copy()

# Shared style for all plots
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Helper function to smooth the data
def smooth(series: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    """Return rolling mean; raw values where window is not yet full."""
    return series.rolling(window, min_periods=1).mean()


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# Total Return per Episode
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ep_df["episode"], ep_df["total_return"], alpha=0.25, color="steelblue", linewidth=0.8, label="Raw")
ax.plot(ep_df["episode"], smooth(ep_df["total_return"]), color="steelblue", linewidth=2, label=f"Smoothed (w={SMOOTH_WINDOW})")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Return")
ax.set_title("Total Return per Episode")
ax.legend()
save(fig, "01_total_return.png")

# Actor Loss and Critic Loss over Training Steps
# Aggregate per-step losses to per-episode mean (step metrics have one row per step)
loss_cols = ["episode", "actor_loss", "critic_loss"]
loss_ep = (
    st_df[loss_cols]
    .dropna(subset=["actor_loss", "critic_loss"])
    .groupby("episode", as_index=False)
    .mean()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for ax, col, color, title in [
    (axes[0], "actor_loss", "tomato", "Actor Loss"),
    (axes[1], "critic_loss", "mediumseagreen", "Critic Loss"),
]:
    ax.plot(loss_ep["episode"], loss_ep[col], alpha=0.25, color=color, linewidth=0.8)
    ax.plot(loss_ep["episode"], smooth(loss_ep[col]), color=color, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Loss")
    ax.set_title(title)
fig.suptitle("SAC Training Losses (mean per episode)")
fig.tight_layout()
save(fig, "02_sac_losses.png")

# Entropy Temperature (Alpha) over Training
alpha_ep = (
    st_df[["episode", "alpha"]]
    .dropna(subset=["alpha"])
    .groupby("episode", as_index=False)
    .mean()
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(alpha_ep["episode"], alpha_ep["alpha"], color="mediumpurple", linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Alpha (entropy temperature)")
ax.set_title("SAC Entropy Temperature (α) over Training")
save(fig, "03_alpha.png")

# Max Altitude Reached per Episode
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ep_df["episode"], ep_df["max_altitude"] / 1000, alpha=0.25, color="darkorange", linewidth=0.8)
ax.plot(ep_df["episode"], smooth(ep_df["max_altitude"] / 1000), color="darkorange", linewidth=2)
ax.axhline(80, color="black", linestyle="--", linewidth=1, label="Target orbit (80 km)")
ax.set_xlabel("Episode")
ax.set_ylabel("Max Altitude (km)")
ax.set_title("Maximum Altitude Reached per Episode")
ax.legend()
save(fig, "04_max_altitude.png")

# Apoapsis per Episode
fig, ax = plt.subplots(figsize=(8, 4))
apoapsis_km = ep_df["apoapsis"] / 1000
ax.plot(ep_df["episode"], apoapsis_km, alpha=0.25, color="teal", linewidth=0.8)
ax.plot(ep_df["episode"], smooth(apoapsis_km), color="teal", linewidth=2)
ax.axhline(80, color="black", linestyle="--", linewidth=1, label="Target orbit (80 km)")
ax.set_xlabel("Episode")
ax.set_ylabel("Apoapsis (km)")
ax.set_title("Apoapsis per Episode")
ax.legend()
save(fig, "05_apoapsis.png")

# Success Rate (rolling window)
ep_df["success_int"] = ep_df["success"].astype(int)
rolling_success = ep_df["success_int"].rolling(SMOOTH_WINDOW, min_periods=1).mean() * 100

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ep_df["episode"], rolling_success, color="goldenrod", linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel(f"Success Rate (%) — {SMOOTH_WINDOW}-ep window")
ax.set_title("Orbit Achievement Success Rate")
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.set_ylim(0, 105)
save(fig, "06_success_rate.png")

# Termination Reason Distribution
reason_counts = ep_df["termination_reason"].value_counts()

REASON_LABELS = {
    "vessel_destroyed": "Vessel Destroyed",
    "below_ground": "Below Ground",
    "max_steps": "Max Steps",
    "orbit_achieved": "Orbit Achieved",
}
labels = [REASON_LABELS.get(r, r) for r in reason_counts.index]
colors = ["tomato", "sandybrown", "steelblue", "mediumseagreen"][:len(reason_counts)]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, reason_counts.values, color=colors, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, padding=3)
ax.set_ylabel("Number of Episodes")
ax.set_title(f"Termination Reasons (Episodes 0–{MAX_EPISODE})")
ax.tick_params(axis="x", rotation=15)
save(fig, "07_termination_reasons.png")

# Reward Components over Training
reward_cols = [
    "reward_potential",
    "reward_alive",
    "reward_overshoot",
    "reward_orbit_bonus",
    "reward_failure_penalty",
    "reward_ground_penalty",
]
reward_colors = ["steelblue", "mediumseagreen", "darkorange", "goldenrod", "tomato", "saddlebrown"]

# Aggregate mean reward components per episode
rew_ep = (
    st_df[["episode"] + reward_cols]
    .groupby("episode", as_index=False)
    .mean()
)

fig, ax = plt.subplots(figsize=(10, 5))
for col, color in zip(reward_cols, reward_colors):
    label = col.replace("reward_", "").replace("_", " ").title()
    smoothed = smooth(rew_ep[col])
    ax.plot(rew_ep["episode"], smoothed, label=label, color=color, linewidth=1.8)

ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax.set_xlabel("Episode")
ax.set_ylabel("Mean Reward Component (per step)")
ax.set_title("Reward Component Breakdown over Training")
ax.legend(loc="upper left", fontsize=8)
save(fig, "08_reward_components.png")

#Fuel Remaining per Episode
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ep_df["episode"], ep_df["fuel_remaining_frac"] * 100, alpha=0.25, color="cornflowerblue", linewidth=0.8)
ax.plot(ep_df["episode"], smooth(ep_df["fuel_remaining_frac"] * 100), color="cornflowerblue", linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Fuel Remaining (%)")
ax.set_title("Fuel Remaining at Episode End")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
save(fig, "09_fuel_remaining.png")

#Episode Length over Training
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ep_df["episode"], ep_df["episode_steps"], alpha=0.25, color="slategray", linewidth=0.8)
ax.plot(ep_df["episode"], smooth(ep_df["episode_steps"]), color="slategray", linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Steps per Episode")
ax.set_title("Episode Length over Training")
save(fig, "10_episode_length.png")

print(f"\nAll figures saved to {OUT_DIR}/")
