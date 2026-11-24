"""Plot logged training loss, retrospective validation loss, and solver validation."""
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / "experiments/cluster_selection/results/mask_sudoku.FGN32oF0"


def main():
    destination = HERE / "figures"
    destination.mkdir(exist_ok=False)
    outputs = [destination / name for name in ("learning_curves.png", "learning_curves.svg", "curves.json", "README.txt")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Plot outputs already exist; preserve the previous figures")
    paths = sorted((RUN / "train").glob("metrics_*.json"))
    training = [json.loads(path.read_text()) for path in paths]
    validation_path = HERE / "validation_loss.json"
    validation = json.loads(validation_path.read_text())
    initial_path = RUN / "train/validation_0000.json"
    initial = json.loads(initial_path.read_text())
    selection_path = RUN / "train/selection.json"
    selection = json.loads(selection_path.read_text())
    selected = int(Path(selection["checkpoint"]).stem.rsplit("_", 1)[1])
    epochs = [row["epoch"] for row in training]
    assert epochs == list(range(1, 31))
    val_rows = validation["rows"]
    assert [row["epoch"] for row in val_rows] == [0, 5, 10, 15, 20, 25, 30]
    solving = [(0, initial["trained"]["total_nodes"])] + [
        (row["epoch"], row["validation"]["total_nodes"])
        for row in training if "validation" in row]
    assert all(nodes is not None for _, nodes in solving)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, (loss_ax, nodes_ax) = plt.subplots(1, 2, figsize=(12.8, 5.6))
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.29, wspace=.28)
    fig.suptitle("Learning cluster masks for Sudoku", fontsize=19, fontweight="bold", y=.97)
    fig.text(.5, .89, "30 epochs from random initialization  |  fixed validation puzzles  |  lower is better",
             ha="center", color="#475569", fontsize=11)
    loss_ax.plot(epochs, [row["metrics"]["loss"] for row in training],
                 color="#2563eb", lw=2.5, label="Training: online batch mean")
    loss_ax.plot([row["epoch"] for row in val_rows], [row["metrics"]["loss"] for row in val_rows],
                 color="#d97706", marker="o", lw=2.5, label="Validation: saved checkpoints")
    loss_ax.set(title="Training and validation loss", xlabel="Epoch", ylabel="Supervised imitation loss")
    loss_ax.set_ylim(bottom=0)
    loss_ax.legend(loc="upper right", frameon=False, fontsize=10)
    val_losses = {row["epoch"]: row["metrics"]["loss"] for row in val_rows}
    loss_ax.text(.98, .69, "Epoch 25 to 30\n"
                 f"Training: {training[24]['metrics']['loss']:.2f} to {training[29]['metrics']['loss']:.2f}\n"
                 f"Validation: {val_losses[25]:.2f} to {val_losses[30]:.2f}",
                 transform=loss_ax.transAxes, ha="right", va="top", color="#475569", fontsize=10)
    nodes_ax.plot(*zip(*solving), color="#7c3aed", marker="o", lw=2.5, label="Learned model, 1 anchor")
    for mode, label, color in (("plain", "Plain MAC", "#64748b"),
                               ("heuristic", "Structural heuristic, allowance 64", "#15803d")):
        nodes = initial[mode]["total_nodes"]
        nodes_ax.axhline(nodes, color=color, ls="--", lw=1.5)
        nodes_ax.text(.98, nodes + 5, f"{label}: {nodes}", transform=nodes_ax.get_yaxis_transform(),
                      ha="right", va="bottom", color=color, fontsize=9)
    best_nodes = dict(solving)[selected]
    nodes_ax.annotate(f"Selected: {best_nodes}", (selected, best_nodes),
                      xytext=(-22, -24), textcoords="offset points", ha="right",
                      fontsize=10, color="#5b21b6", arrowprops={"arrowstyle": "-", "color": "#7c3aed"})
    nodes_ax.annotate(str(solving[-1][1]), solving[-1], xytext=(-6, 9),
                      textcoords="offset points", ha="right", color="#5b21b6", fontsize=10)
    nodes_ax.set(title="Validation solving performance", xlabel="Epoch", ylabel="DFS nodes across 55 puzzles")
    nodes_ax.set_ylim(265, 520)
    nodes_ax.legend(loc="upper right", frameon=False, fontsize=9)
    for ax in (loss_ax, nodes_ax):
        ax.set_xlim(-.5, 31)
        ax.set_xticks(range(0, 31, 5))
        ax.axvline(selected, color="#94a3b8", ls=":", lw=1, zorder=0)
        ax.grid(axis="y", alpha=.18)
    fig.text(.075, .145, f"Retrospective validation loss: {validation['samples']} heuristic-labelled clusters from "
             f"{validation['labelled_puzzles']}/55 reserved puzzles.", fontsize=10, color="#475569")
    fig.text(.075, .10, "Loss uses teacher anchors; training loss is an online mini-batch mean. Saved weights were not updated.",
             fontsize=10, color="#475569")
    fig.text(.075, .055, "Solver curve: logged greedy, single-anchor MAC; allowance 64. Validation split only; no final test data.",
             fontsize=10, color="#475569")
    fig.savefig(outputs[0], dpi=180, facecolor="white")
    svg = StringIO()
    fig.savefig(svg, format="svg", facecolor="white")
    with outputs[1].open("x") as stream:
        stream.write("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n")
    plt.close(fig)
    result = dict(training=[dict(epoch=row["epoch"], **row["metrics"]) for row in training],
                  validation_loss=val_rows, validation_solver=[dict(epoch=e, nodes=n) for e, n in solving],
                  plain_mac_nodes=initial["plain"]["total_nodes"],
                  heuristic_nodes=initial["heuristic"]["total_nodes"], selected_epoch=selected,
                  sources_sha256={str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                  for path in [*paths, initial_path, selection_path, validation_path]},
                  note="Training loss averages batches during parameter updates; validation loss scores fixed labels at saved epoch endpoints.")
    with outputs[2].open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report = ["Training and validation curves for direct supervised cluster-mask learning",
              "", f"Run: {RUN.relative_to(ROOT)}", "Status: all 30 epochs completed; Condor job 7974 returned 0.",
              f"Selected checkpoint: epoch {selected}, chosen by the original validation solver results.",
              "", "epoch   logged training loss   retrospective validation loss   validation DFS nodes"]
    logged = {row['epoch']: row['metrics']['loss'] for row in training}
    for row in val_rows:
        epoch = row['epoch']
        training_text = f"{logged[epoch]:.4f}" if epoch in logged else "not logged"
        report.append(f"{epoch:5d}   {training_text:>20}   {row['metrics']['loss']:>29.4f}   {dict(solving)[epoch]:>20d}")
    report.extend(["", "Loss = anchor loss + core-membership loss + open-variable loss + 0.25 * count loss.",
                   "Training losses are original online batch averages during parameter updates.",
                   "Validation loss was not recorded during training; it was recomputed afterward on saved checkpoints.",
                   f"Fixed validation labels: {validation['samples']} useful, noncontradictory heuristic applications from "
                   f"{validation['labelled_puzzles']} of the 55 reserved puzzles; equal weight per application.",
                   "A fixed uniformly sampled actually-pruned teacher anchor is supplied for each example at every checkpoint.",
                   "Validation labels come from heuristic-64 root and DFS applications; the training collection used allowance 256 and mostly root states.",
                   "The raw train/validation gap therefore also reflects a different teacher/state collection distribution.",
                   "Validation loss weights applications equally; puzzle 21 supplies 113/339 labels (one third).",
                   "Solver validation uses the model's greedy anchor and generated cluster; its nodes are not a loss.",
                   "The validation puzzles were used for checkpoint selection; this is not a final test evaluation.",
                   "No training, gradients, checkpoint modifications, or changes to the original training run.",
                   "", "learning_curves.png / .svg: figure. curves.json: plotted values and source hashes.",
                   "../validation_loss_protocol.json and ../evaluation_labels/: fixed evaluation-only labels and provenance.",
                   "These validation labels must never be added to training data."])
    with outputs[3].open("x") as stream:
        stream.write("\n".join(report) + "\n")
    print(outputs[0])


if __name__ == "__main__":
    main()
