import re
import numpy as np
import matplotlib.pyplot as plt

def parse_log(path):
    pattern = re.compile(
        r"Episode\s+(\d+)\s*\|"
        r"\s*Reward:\s*([\d.]+)\s*\|"
        r"\s*Steps:\s*([\d.]+)\s*\|"
        r"\s*Max X:\s*([\d.]+)\s*\|"
        r"\s*Loss:\s*([\d.]+)"
    )
    episodes, rewards, steps, max_x, losses = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
                steps.append(float(m.group(3)))
                max_x.append(float(m.group(4)))
                losses.append(float(m.group(5)))
    return (
        np.array(episodes),
        np.array(rewards),
        np.array(steps),
        np.array(max_x),
        np.array(losses),
    )

def rolling_avg(data, window=50):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")

episodes, rewards, steps, max_x, losses = parse_log("poster_run.txt")

metrics = [
    ("Reward",  rewards, "tab:blue"),
    ("Steps",   steps,   "tab:orange"),
    ("Max X Position",   max_x,   "tab:green"),
]

WINDOW = 50

fig, axes = plt.subplots(3, 1, figsize=(20, 26))
# fig.suptitle("Rainbow DQN Training Results", fontsize=30, fontweight="bold")

for ax, (title, data, color) in zip(axes.flat, metrics):
    avg = data.mean()
    roll = rolling_avg(data, WINDOW)
    roll_ep = episodes[WINDOW - 1:]

    ax.plot(episodes, data, color=color, alpha=0.35, linewidth=1.5, label="per episode")
    ax.plot(roll_ep, roll, color=color, linewidth=4, label=f"rolling avg (w={WINDOW})")
    ax.axhline(avg, color="black", linewidth=3, linestyle="--", label=f"mean = {avg:.2f}")

    ax.set_title(title, fontsize=35, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=30)
    ax.set_ylabel(title, fontsize=30)
    ax.tick_params(axis="both", labelsize=30)
    ax.legend(fontsize=25)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("poster_run_plot.png", dpi=200)
print("Saved poster_run_plot.png")
plt.show()
