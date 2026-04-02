"""
Run this in a separate terminal while training to visualize progress:
  /home/sam/Github/Project-SMW/venv/bin/python3 plot_log.py
"""
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'results', 'default', 'train_log.csv')
OUT_PATH = os.path.join(SCRIPT_DIR, 'results', 'default', 'training_progress.png')


def read_log(path):
    rows = {'episode': [], 'eval_start': [], 'eval_end': []}
    if not os.path.exists(path):
        return rows
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row['phase']
            if phase in rows:
                rows[phase].append(row)
    return rows


def plot(rows):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Rainbow DQN — SMW Training Log')

    # --- Steps vs time (detect freezes: big gaps = hang) ---
    ax = axes[0]
    if rows['episode']:
        steps = [int(r['step']) for r in rows['episode']]
        times = [datetime.fromisoformat(r['timestamp']) for r in rows['episode']]
        ax.plot(times, steps, color='steelblue')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylabel('Step')
        ax.set_title('Steps over time (gaps = freeze)')
        ax.grid(True, alpha=0.3)
        for r in rows['eval_start']:
            t = datetime.fromisoformat(r['timestamp'])
            ax.axvline(t, color='orange', linestyle='--', alpha=0.7)
        for r in rows['eval_end']:
            t = datetime.fromisoformat(r['timestamp'])
            ax.axvline(t, color='green', linestyle='--', alpha=0.7)

    # --- Episode reward ---
    ax = axes[1]
    if rows['episode']:
        steps = [int(r['step']) for r in rows['episode'] if r['reward'] != '']
        rewards = [float(r['reward']) for r in rows['episode'] if r['reward'] != '']
        ax.plot(steps, rewards, color='coral', alpha=0.5, linewidth=0.8)
        # Smoothed moving average
        if len(rewards) >= 20:
            window = max(1, len(rewards) // 20)
            smoothed = [sum(rewards[max(0,i-window):i+1]) / min(i+1, window+1) for i in range(len(rewards))]
            ax.plot(steps, smoothed, color='red', linewidth=2, label='smoothed')
            ax.legend()
        ax.set_xlabel('Step')
        ax.set_ylabel('Episode reward (x-pos delta)')
        ax.set_title('Episode reward (x-position gained per episode)')
        ax.grid(True, alpha=0.3)

    # --- Eval average reward ---
    ax = axes[2]
    if rows['eval_end']:
        eval_steps = [int(r['step']) for r in rows['eval_end']]
        eval_rewards = [float(r['reward']) for r in rows['eval_end']]
        ax.plot(eval_steps, eval_rewards, marker='o', color='mediumseagreen')
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Reward')
        ax.set_title('Evaluation average reward')
        ax.grid(True, alpha=0.3)

        # Show eval duration
        for s, e in zip(rows['eval_start'], rows['eval_end']):
            t_start = datetime.fromisoformat(s['timestamp'])
            t_end = datetime.fromisoformat(e['timestamp'])
            duration = (t_end - t_start).total_seconds()
            print(f"  Eval at step {s['step']}: {duration:.1f}s")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=120)
    plt.show()
    print(f"Saved to {OUT_PATH}")


if __name__ == '__main__':
    rows = read_log(LOG_PATH)
    ep_count = len(rows['episode'])
    last_step = rows['episode'][-1]['step'] if rows['episode'] else 'N/A'
    last_time = rows['episode'][-1]['timestamp'] if rows['episode'] else 'N/A'
    print(f"Episodes logged: {ep_count} | Last step: {last_step} | Last timestamp: {last_time}")
    print(f"Eval starts: {len(rows['eval_start'])} | Eval ends: {len(rows['eval_end'])}")
    if len(rows['eval_start']) > len(rows['eval_end']):
        print("  *** eval_start has no matching eval_end — likely frozen inside test() ***")
    plot(rows)