# Completed verification

- 50 unique main runs: two architectures × five dataset sizes × five seeds.
- Every run completed its target updates and generated 1,000 valid 14×16 tile arrays: 50,000 outputs total.
- Architecture pairs have identical ordered training-level lists. Within each seed, level subsets are nested. No training level is in the fixed five-level holdout.
- All saved generated-to-training nearest Hamming distances were recomputed from arrays and matched exactly. Boundary cases at distances 0, 11, and 12 passed.
- The local reachability proxy passed flat-ground and fully-blocked examples. It is not a validated Mario simulator.
- Main job 35873514 completed in 916 seconds. Pilot and failed validation consumed 96 and 20 seconds respectively; cancelled queued job consumed zero. Total: 1,032 seconds (17:12), below 2,700 seconds authorized. See resources.tsv.
- The final ALT-template report compiled to three pages and every page was visually inspected. Figures and table derive from results/summary.json and the saved arrays.
- No evidence of convergence is claimed. Short budgets, different capacities, common patterns, approximate reachability, and fixed-update exposure differences limit interpretation.
- The private GitHub repository is sujalstha/MarioGPT. Invitation to mlciv was sent with the supported personal-repository collaborator permission; acceptance was pending at verification.
- Student review and the required meeting with Dr. Cao remain outstanding.
