Training and validation curves for direct supervised cluster-mask learning

Run: experiments/cluster_selection/results/mask_sudoku.FGN32oF0
Status: all 30 epochs completed; Condor job 7974 returned 0.
Selected checkpoint: epoch 25, chosen by the original validation solver results.

epoch   logged training loss   retrospective validation loss   validation DFS nodes
    0             not logged                         12.0373                    479
    5                 6.0928                          6.2604                    479
   10                 4.6544                          5.1656                    442
   15                 4.2277                          4.6938                    388
   20                 4.0226                          4.7587                    396
   25                 3.9016                          4.4835                    364
   30                 3.8266                          4.5053                    401

Loss = anchor loss + core-membership loss + open-variable loss + 0.25 * count loss.
Training losses are original online batch averages during parameter updates.
Validation loss was not recorded during training; it was recomputed afterward on saved checkpoints.
Fixed validation labels: 339 useful, noncontradictory heuristic applications from 45 of the 55 reserved puzzles; equal weight per application.
A fixed uniformly sampled actually-pruned teacher anchor is supplied for each example at every checkpoint.
Validation labels come from heuristic-64 root and DFS applications; the training collection used allowance 256 and mostly root states.
The raw train/validation gap therefore also reflects a different teacher/state collection distribution.
Validation loss weights applications equally; puzzle 21 supplies 113/339 labels (one third).
Solver validation uses the model's greedy anchor and generated cluster; its nodes are not a loss.
The validation puzzles were used for checkpoint selection; this is not a final test evaluation.
No training, gradients, checkpoint modifications, or changes to the original training run.

learning_curves.png / .svg: figure. curves.json: plotted values and source hashes.
../validation_loss_protocol.json and ../evaluation_labels/: fixed evaluation-only labels and provenance.
These validation labels must never be added to training data.
