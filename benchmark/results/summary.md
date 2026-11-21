# ShelfSage benchmark: final results

Tasks: benchmark/tasks.json (42). Baseline: 1 run (deterministic). LLM: 3 repeats executed, 3 at plan level. Cells: success rate (range over repeats) [95 % Wilson interval], successes / counted runs.

| category | baseline plan | baseline executed | LLM plan | LLM executed |
|---|---|---|---|---|
| single | 100 % [68-100], 8/8 | 100 % [68-100], 8/8 | 100 % (100-100) [86-100], 24/24 | 100 % (100-100) [86-100], 24/24 |
| multi | 17 % [3-56], 1/6 | 17 % [3-56], 1/6 | 89 % (83-100) [67-97], 16/18 | 78 % (67-83) [55-91], 14/18 |
| phrasing | 0 % [0-32], 0/8 | 0 % [0-32], 0/8 | 92 % (88-100) [74-98], 22/24 | 92 % (88-100) [74-98], 22/24 |
| spatial | 0 % [0-32], 0/8 | 0 % [0-32], 0/8 | 100 % (100-100) [86-100], 24/24 | 88 % (88-88) [69-96], 21/24 |
| query | 50 % [15-85], 2/4 | 50 % [15-85], 2/4 | 100 % (100-100) [76-100], 12/12 | 100 % (100-100) [76-100], 12/12 |
| infeasible | 100 % [44-100], 3/3 | 100 % [44-100], 3/3 | 100 % (100-100) [70-100], 9/9 | 100 % (100-100) [70-100], 9/9 |
| failure | - | 60 % [23-88], 3/5 | - | 93 % (80-100) [70-99], 14/15 |
| all | 38 % [24-54], 14/37 | 40 % [27-56], 17/42 | 96 % (95-100) [91-99], 107/111 | 92 % (90-95) [86-96], 116/126 |

Paired per task, executed (LLM majority of its repeats): both succeed 17, only the baseline 0, only the LLM 22, neither 3; exact McNemar p = 4.8e-07.
LLM tasks whose outcome changed between repeats: 3: M04 oxo, P06 oxx, F02 oox

LLM planning: 111 requests, latency median 2.6 s, 90th percentile 4.3 s, max 25.3 s; validator retries in 0 of 111; ~6389 input / 741 output tokens per command.
Simulated time per successful task, baseline: median 121 s, range 78-175 s (n = 12).
Simulated time per successful task, llm: median 93 s, range 75-271 s (n = 95).

Plots: plots/success_by_category.png, plots/failure_recovery.png, plots/llm_latency.png
