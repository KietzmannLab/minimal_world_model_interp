
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from setup.data_processing import process_data
from setup.model import GP_model
from setup.config import DEVICE, GRAPHS_DIR, PALETTE
import seaborn as sns

def save_plot(plt, graphs_dir, title):
    import os
    from datetime import datetime

    # Replace spaces and other unsafe filename characters
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")

    os.makedirs(graphs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plot_path = os.path.join(graphs_dir, f"{timestamp}_{safe_title}.pdf")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved to: {plot_path}")

    plt.show()
    return plot_path


def plot_token_world(world, token_colors, title=None, connections=None, save=False, graphs_dir=None ):
    plt.figure(figsize=(8, 8))
    min_pos, max_pos = world.grid
    ax = plt.gca()

    # Plot grid lines
    step = 1.0  # Grid granularity; can be adjusted
    ticks = np.arange(min_pos, max_pos + step, step)
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Invert y-axis so (0,0) is top-left
    # ax.invert_yaxis()

    # Plot tokens
    xs, ys = [], []
    for token in world.tokens:
        y, x = token.coordinates
        xs.append(x)
        ys.append(y)
        plt.scatter(x, y, color=token_colors[token], alpha=0.7, s=500, edgecolors="black")
        plt.text(
            x,
            y,
            token.label,
            fontsize=14,
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
        )

    if connections is not None:
        # Draw connections
        for token1, token2 in connections:
            y1, x1 = token1.coordinates
            y2, x2 = token2.coordinates
            plt.plot([x1, x2], [y1, y2], linestyle="--", color="gray", alpha=0.6)

    # Axis limits with slight padding
    pad = 0.25
    plt.xlim(min_pos - pad, max_pos + pad)
    plt.ylim(max_pos + pad, min_pos - pad)

    plt.title("Token World Grid")
    if save: 
        save_plot(plt, graphs_dir, "world_grid")



    plt.fill_between(positions, lower_bound, upper_bound, alpha=0.2, label="CI 95%", color=c2)


    if metric == "loss":
        plt.ylim(0, 10)
    else: 
         plt.ylim(0, 1)

    plt.title(title,  pad=15) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel,  labelpad=15) 
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()

    save_name = title.replace(" ", "_")
    save_plot(plt, graphs_dir, f"{save_name}")

def pairwise_embedding_correlation(world_seqs, model, model_type, device, checkpoints_dir, graphs_dir):
    """pair-wise comparison of one token embedding with its next occurence"""
    token_differences = []
    for world, seqs in world_seqs.items():
        # Prepare a single world_world_seq for this sequence
        world_world_seq = {world: seqs}
        emb_comparison = compare_embeddings(
            model, model_type, world_world_seq, device, checkpoints_dir
        )
        token_differences.append(emb_comparison)

    max_steps = max(
        max(len(correlations) for correlations in seq.values()) for seq in token_differences
    )
    # Initialize storage for averaging
    step_correlations = {step: [] for step in range(1, max_steps + 1)}

    # Aggregate correlations at each step across all sequences
    for emb_comparison in token_differences:  # Loop over 1000 sequences
        for token, correlations in emb_comparison.items():
            for step, corr in enumerate(correlations, start=1):
                step_correlations[step].append(corr)

    # Compute means and standard deviations
    steps = sorted(step_correlations.keys())
    means = [np.mean(step_correlations[step]) for step in steps]
    std_devs = [np.std(step_correlations[step]) for step in steps]

    # Plot mean line with shaded standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(
        steps, means, marker="o", linestyle="-", linewidth=2, alpha=0.8, label="Mean Correlation"
    )
    plt.fill_between(
        steps,
        np.array(means) - np.array(std_devs),
        np.array(means) + np.array(std_devs),
        alpha=0.2,
        label="±1 Std Dev",
    )
    plt.title("Average correlation of token embeddings pairs")
    plt.xlabel("Token Occurrence Step")
    plt.ylabel("Average Pearson Correlation with next ocurrence")
    plt.xticks(range(1, max_steps + 1))
    plt.ylim(-1, 1)  # Pearson correlation ranges from -1 to 1
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_plot(graphs_dir, "pairwise_embedding_correlation.png")

def compare_embeddings(model, model_type, world_seq, device, checkpoints_dir):
    tokens, directions, targets = process_data(world_seq, model_type=model_type)
    # tokens, directions = prepare_alternating_data(tokens, directions)

    world, (token_seq, direcionts_seq) = next(iter(world_seq.items()))

    tokens = tokens.to(device)
    directions = directions.to(device)
    model.to(device)

    with torch.no_grad():
        rnn_out, hidden_states = model(tokens, directions)
        # activations = activations[:, 1::2, :]
        rnn_out = rnn_out.reshape(-1, rnn_out.size(-1))

    # Ensure lengths match
    num_activations = rnn_out.shape[0]
    token_seq = token_seq[:num_activations]  # Truncate token_seq if needed

    token_activations = {}
    token_differences = {}  # Store per-token step-wise correlation

    for i, token in enumerate(token_seq):
        if token not in token_activations:
            token_activations[token] = []
            token_differences[token] = []  # Initialize per-token list
        token_activations[token].append(rnn_out[i].cpu().numpy())

    for token, acts in token_activations.items():
        for i in range(len(acts) - 1):
            correlation = pearsonr(acts[i], acts[i + 1])[0]  # Compare consecutive steps
            token_differences[token].append(correlation)  # Store per-token correlations

    return token_differences

def compute_mean_and_ci(values, confidence=0.95):

    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, 0.0  # Degenerate case
    se = np.std(values, ddof=1) / np.sqrt(n)
    ci = se * t.ppf((1 + confidence) / 2., n-1)
    return mean, ci




def plot_decay_metric( mean_dict, ci_dict, *, pre_steps, gap_steps, reintro_step=None, title, ylabel, save_path=GRAPHS_DIR):
    """
    Plot metric over time with decay phases highlighted.

    Parameters
    ----------
    mean_dict : dict[int, float]
        Mean metric per timestep.
    ci_dict : dict[int, float]
        CI half-width per timestep.
    pre_steps : int
        Length of pre-exposure phase.
    gap_steps : int
        Length of withdrawal phase.
    reintro_step : int | None
        Timestep of reintroduction (vertical line).
        If None, inferred as pre_steps + gap_steps.
    """

    timesteps = np.array(sorted(mean_dict.keys()))
    means = np.array([mean_dict[t] for t in timesteps])
    cis   = np.array([ci_dict[t]   for t in timesteps])

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Main curve ---
    ax.plot(timesteps, means, linewidth=2.5,  label="Mean")

    ax.fill_between(
        timesteps,
        means - cis,
        means + cis,
        alpha=0.3,
        label="95% CI",
     
    )

    # --- Phase boundaries ---
    pre_start = timesteps[0]
    pre_end   = pre_steps
    gap_end   = pre_steps + gap_steps

    if reintro_step is None:
        reintro_step = gap_end

    # --- Shade pre-exposure ---
    ax.axvspan(
        pre_start,
        pre_end,
        alpha=0.15,
        label="Pre-exposure",
        color=PALETTE[2]
    )

    # --- Reintroduction marker ---
    ax.axvline(
        reintro_step,
        linestyle="--",
        linewidth=1.5,
        zorder=0,      # put behind everything
        alpha=0.50,
        label="Reintroduction",
        color="black"
    )

    # --- Cosmetics ---
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=16,
        handlelength=1.2,
        handletextpad=0.3,
    )
    
    plt.tight_layout()

    if save_path is not None:
        save_plot(plt, save_path, title)
    else:
        plt.show()