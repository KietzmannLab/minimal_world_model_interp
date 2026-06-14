
from datetime import datetime
from pathlib import Path
import pickle
import random
import re
from matplotlib import pyplot as plt
import numpy as np
from regex import F
import torch
import seaborn as sns
from setup.data_processing import process_data

def create_run_folder(run_name: str | None = None, base_dir: str | Path = "outputs") -> Path:
    """
    Create a folder for a new run under base_dir and return the Path.
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path(base_dir)
    run_dir = (base_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created run directory: {run_dir}")
    return run_dir

_EPOCH_RE = re.compile(r"epoch_(\d+)\.pth$")


def find_latest_checkpoint(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    ckpts = []
    for p in run_dir.glob("epoch_*.pth"):
        m = _EPOCH_RE.search(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))

    if not ckpts:
        raise FileNotFoundError(f"No epoch_*.pth checkpoints found in {run_dir}")

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

def plot_token_world(world, token_colors, connections=None, save=False, graphs_dir=None ):
    plt.figure(figsize=(6, 6))
    min_pos, max_pos = world.grid
    ax = plt.gca()

    # Plot grid lines
    step = 1.0  # Grid granularity; can be adjusted
    ticks = np.arange(min_pos, max_pos + step, step)
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax.grid(True, linestyle="--", alpha=0.5)



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


def load_world_sequences(data_path="gaze_sequences.pkl", n_worlds=500):
    with open(data_path, "rb") as f:
        sequences = pickle.load(f)
    return dict(list(sequences.items())[:n_worlds])


def load_model(model, device, checkpoints_dir, checkpoint_file=None):
    checkpoints_dir = Path(checkpoints_dir)

    if checkpoint_file is None:
        ckpts = sorted(checkpoints_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            raise FileNotFoundError(f"No .pth checkpoints found in {checkpoints_dir.resolve()}")
        checkpoint_path = ckpts[-1]  # latest by mtime
    else:
        checkpoint_path = checkpoints_dir / checkpoint_file
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

    print(f"loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    torch.set_grad_enabled(False)
    return model


def entropy_token(model, world_seq):
    tokens_tensors, directions_tensors, targets = process_data(world_seq, model_type, DEVICE)
    world = next(iter(world_seq.keys())) 

    tokens = world.tokens
    hidden_pos = None
    hidden_token = None
    for token in tokens:
        if token.hidden:
            hidden_token = token

    entropies = []

    if tokens_tensors.dim() == 2:
        tokens_tensors = tokens_tensors.unsqueeze(0)        # → [1, T, D]
    if directions_tensors.dim() == 2:
        directions_tensors = directions_tensors.unsqueeze(0)  # → [1, T, 2]

    with torch.no_grad():
        logits = model(tokens_tensors, directions_tensors, return_all_activations=False)

    # Iterate over the sequence length.
    for pos in range(logits.shape[1]):
        # Get the logits for the current position.
        pos_logits = logits[0, pos, :]
        # Compute the probability distribution.
        pos_probs = F.softmax(pos_logits, dim=-1)
        # Compute the entropy: - sum_i p_i * log(p_i)
        entropy = -torch.sum(pos_probs * torch.log(pos_probs + 1e-10))
        entropies.append(entropy.item())

        if hidden_token is not None:
            # find the first position of the hidden token
            if hidden_pos is None and torch.all(tokens_tensors[:, pos, :] == torch.tensor(hidden_token.one_hot_vector, dtype=tokens_tensors[:, pos, :].dtype)):
                hidden_pos = pos - 1

    return entropies

def loss_token(model, world_seq,device):
    # Load and prepare data
    tokens, directions, targets = process_data(world_seq)

    # Move data to device
    tokens = tokens.to(device)
    directions = directions.to(device)
    targets = targets.to(device)
    model = model.to(device)

    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)        # → [1, T, D]
    if directions.dim() == 2:
        directions = directions.unsqueeze(0)  # → [1, T, 2]

    with torch.no_grad():
        # Forward pass
        rnn_out = model(tokens, directions)
        # logits = logits[:, 1::2, :]  # Only keep logits after direction token is given
        rnn_out = rnn_out.reshape(-1, rnn_out.size(-1))
        targets = targets.reshape(-1)

        # Compute per-token loss
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # print to see single value or vector
        per_token_loss = criterion(rnn_out, targets)

    return per_token_loss.cpu().numpy()  # Convert to NumPy for easy processing

def accuracy_token(model, world_seq, device):
    # Load and prepare data
    tokens, directions, targets = process_data(world_seq)

    # Move data to device
    tokens = tokens.to(device)
    directions = directions.to(device)
    targets = targets.to(device)
    model = model.to(device)

    with torch.no_grad():
        rnn_out = model(tokens, directions)
        rnn_out = rnn_out.reshape(-1, rnn_out.size(-1))
        targets = targets.reshape(-1)

        predictions = torch.argmax(rnn_out, dim=-1)

        # Create a binary vector: 1 if correct, 0 otherwise
        per_token_accuracy = predictions.eq(targets).float()

    return per_token_accuracy.cpu().numpy()

def average_metrics(measurements):
    """
    averages over a group of sequences the accuracy or loss per token.
    Returns dicts: means, stds, ns per position.
    """
    grouped = {}
    for seq_measures in measurements:
        for pos, value in enumerate(seq_measures):
            if pos not in grouped:
                grouped[pos] = []
            grouped[pos].append(value)

    mean_losses = {}
    std_losses  = {}
    ns          = {}
    
    for pos, values in grouped.items():
        arr = np.array(values)
        mean_losses[pos] = arr.mean()
        std_losses[pos]  = arr.std(ddof=1)
        ns[pos]          = len(arr)

    return mean_losses, std_losses, ns




def plot_distribution(means, ci_values, graphs_dir, title="Metric Distribution",
                    xlabel="Token Position", ylabel="Metric Value", metric="loss"):
    
    positions = sorted(means.keys())
    means = [means[pos] for pos in positions]

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", 8)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c2 = palette[0]

    plt.plot(positions, means,linewidth=3.5,  label="Mean", color=c2)

    # if metric.lower != "accuracy":
    error_interval = [ci_values[pos] for pos in positions]

    # Shade the area between mean-ci and mean+ci
    lower_bound = [m - s for m, s in zip(means, error_interval)]
    upper_bound = [m + s for m, s in zip(means, error_interval)]

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