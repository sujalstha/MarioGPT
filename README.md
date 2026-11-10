# Do Generative Level Models Copy Their Training Data?

CS 4713, Fall 2026 — Sujal Shrestha and Jackson McKay.

This is a budget-limited adaptation study, not an exact reproduction of MarioGPT or Lee and Simo-Serra. The main grid is 2 architectures × 5 training-set sizes (3, 6, 9, 12, 15) × 5 seeds. Each run generates 1,000 14×16 slices. Results and limitations are in the accompanying LaTeX report.

## Data

Clone https://github.com/TheVGLC/TheVGLC at revision `0e86a8f31f20ecad4eaa5741ff061af88767f7fb`. Copy `.txt` maps from `Super Mario Bros/Processed` and `Super Mario Bros 2 (Japan)/Processed` into `data/`. Retain the upstream license. Split whole levels before making stride-1 slices; the fixed split seed is 2026. Five levels are held out. Per-run seed controls the nested level subset and training/generation RNG. The manifest and per-run JSON record exact level IDs.

## Environment and reproduction

On OSCER, load `PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`, create a virtual environment with `--system-site-packages`, and install `requirements.txt`. `prepare.sh` downloads DistilGPT2 weights before GPU allocation. `sbatch job.sh 2` runs the main grid and skips runs with completed result JSON. Adjust account/partition only to match your own allocation. Never train on a login node.

The Transformer retains pretrained DistilGPT2 blocks but replaces its language-token embedding/output vocabulary with tile IDs plus a start token. It has no text prompts or novelty search. It trains for 300 updates (batch 16, AdamW, learning rate 5e-5, zero weight decay), then samples with temperature 1 and no top-k truncation. Explicit attention masks include the start token. The tiny diffusion model is a single-downsample U-Net with a skip connection (32/64/32 channels), scalar time channel, 100 linear-beta DDPM steps (0.0001 to 0.02), epsilon-prediction MSE, and argmax decoding. It trains for 1,500 updates (batch 16, Adam, learning rate 1e-3). It omits the original diffusion paper's attention modules. These deliberately short budgets are a limitation, not evidence of convergence.

`run.py` writes per-run JSON, compressed arrays (outputs, training slices, distances, loss history), and representative checkpoints for the three-level seed-0 runs. Checkpoints are excluded from Git. `evaluate.py` recomputes distances, checks saved measurements, computes the reachability proxy, and creates vector PDF figures. Run `python evaluate.py` after downloading `results/` without checkpoints. Metric and simple reachability checks are available through `python -c 'from evaluate import checks; checks()'`.

## Interpretation

A near-copy differs from its nearest training slice in at most 11 of 224 tiles. Exact copies differ in zero. Compare against held-out real levels because shared structures and a growing search corpus can increase matches. Report sample SD across seeds; outputs from one model are not independent training runs.

The reachability proxy searches landing positions supported by solid tiles and discretized jump arcs of up to four tiles horizontally, four up/down, and arc heights 1–4. Empty space and coins are passable; enemies block the path. It is a point-agent local-slice proxy with approximate collision sampling, not Mario physics or an A* full-level playability score. Zero non-copied samples yields an unavailable conditional rate.

## Resource and assistance record

The user authorized at most 45 allocated GPU-minutes, with one GPU at a time. Initial pilot job `35873371` completed in 1:36. It used only base-game levels and exposed a start-token attention-mask warning; its results are excluded from the main analysis. Main job `35873430` has a 30-minute hard limit. Thus these two jobs cannot exceed 31:36 allocated GPU-minutes. Setup/downloads use no GPU allocation.

Codex prepared and executed the code, analysis, figures, and report at Sujal's direction. Individual student technical contributions have not been reported and must not be inferred from authorship. The students must review the artifacts and complete the required 15-minute meeting with Dr. Cao. The repository is private; `mlciv` has been invited as a collaborator.

## Sources

- MarioGPT: https://arxiv.org/abs/2302.05981 and https://github.com/shyamsn97/mario-gpt
- Lee and Simo-Serra (2023): https://esslab.jp/~ess/publications/LeeMVA2023.pdf
- ALT template: https://www.engineering.upenn.edu/~surbhig/alt2027-template.zip

No OSCER credentials, local login documents, or model weights belong in this repository.
