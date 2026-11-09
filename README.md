# Diffusion Restoration Fidelity Audit

Experiment code and archived numerical results for When Does DiffBIR Beat the Input? A Multi-Objective Audit Across Gaussian Blur Severities.

- notebooks/: marimo experiment entry points. Some require authorized ImageNet data, upstream model weights and frozen run inputs.
- experiments/: experiment modules and their local imports.
- analysis/: scripts used for numerical analysis. Some retain paths to the original workspace or require original run archives.
- results/: scores, paired effects, frontiers, source inventories, protocol and frozen choices. Evidence-audit table filenames refer to the revision in which they were generated.

Start with results/input_pareto for frontier tables and results/observed_blur_policy for policy effects. See the notebook headers for input requirements. Raw datasets and checkpoints are obtained separately under their original terms. This bundle is an archival research release. End-to-end clean-checkout execution has not been verified.
