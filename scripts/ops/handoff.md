# scripts/ops/handoff.sh

One-command way to move work between the two LAN machines when one side is at capacity: `server` (this Windows/WSL box, runs the live trading stack) and `dev`.

## Setup (once per machine)

Copy `handoff.hosts.conf.example` to `handoff.hosts.conf` in this same directory and fill in the real `user@lan_ip:port|repo_path|conda_base|conda_env` for both hosts. That file is gitignored — never commit it, this repo is public.

## Commands

```
handoff.sh push   <host> <path...>                          # rsync local path(s) -> host
handoff.sh pull   <host> <path...>                           # rsync host path(s) -> local
handoff.sh launch <host> <job> [--sync <path...>] -- <cmd>   # (optionally sync) + start cmd in background on host, conda env auto-activated
handoff.sh stop   <host> <job>                               # kill it
handoff.sh status <host> [job]                                # RUNNING/STOPPED + recent log
handoff.sh logs   <host> <job> [-f]                            # tail / follow log
```

Paths are relative to the repo root. Jobs live under `tmp/handoff_jobs/<job>/` on the target (`log`, `pid`, the shipped `run.sh` wrapper).

## Typical flow: dev is full, run something on server

```bash
# from dev
bash scripts/ops/handoff.sh launch server optuna_run1 --sync data/ensemble/some_features -- \
  python ensemble/supervised/train_xxx.py --n-trials 200

bash scripts/ops/handoff.sh status server optuna_run1     # or: logs server optuna_run1 -f

# once it's done
bash scripts/ops/handoff.sh pull server data/ensemble/optuna_run1_output
```

Same in reverse (`launch dev ...` run from server) when server is the busy one.

## Notes

- `server` runs `trading_bot.py`, the dashboard, and the shadow loops continuously. Never `wsl --shutdown` it to fix networking — use `netsh interface portproxy` (Windows-side, additive, non-disruptive) instead.
- SSH is key-based (passwordless) in both directions.
- If `launch` seems to pause for a few seconds before returning, that's expected — the launching SSH call is capped with `timeout` and the real state is re-checked over a fresh connection right after. The job itself starts immediately regardless.
