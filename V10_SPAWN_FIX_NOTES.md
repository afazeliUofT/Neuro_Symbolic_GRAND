# V10 spawn/serial-safe generation fix

This update fixes the observed 0-CPU ProcessPool hang during failed-state dataset generation by using multiprocessing spawn rather than fork, recycling child processes with max_tasks_per_child=1, lowering default generation workers to 8, and adding heartbeat logging while waiting for shards. If multiprocessing still stalls on a site, set `data.workers: 1` and rerun the resumable generate stage serially.
