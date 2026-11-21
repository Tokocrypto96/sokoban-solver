# Sokoban Solver

A push-optimal Sokoban solver for the make.fun bounty
[sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121).

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Tokocrypto96/sokoban-solver/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Tokocrypto96/sokoban-solver/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

`main.ts` reads every level under `/input/<id>/level.txt` (XSB format) and
writes the player's moves to `/output/<id>/solution.txt`. It decomposes each
level into its hallway and rooms, then solves every room push-optimally with
an A* search over pushes, guided by a minimum-cost box-to-goal matching
heuristic over precomputed push distances, with static dead-square and
freeze-deadlock pruning. The search budget is counted in node expansions, so
runs are byte-for-byte deterministic. No dependencies; plain Node.js.
