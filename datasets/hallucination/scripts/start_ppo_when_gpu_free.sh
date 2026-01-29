#!/bin/bash
set -euo pipefail

SESSION="ppo_loop"
LOG="ppo_scheduler.log"
CHECK_INTERVAL=30
MEM_THRESHOLD=8000
if [[ -z "${GEMINI_API_KEY:-}" && -f "$HOME/.gemini_api" ]]; then
  source "$HOME/.gemini_api"
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "[$(date)] GEMINI_API_KEY not set; cannot start PPO." >> "$LOG"
  exit 1
fi
COMMAND="cd /root && GEMINI_API_KEY='$GEMINI_API_KEY' PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python custom_ppo.py --num-epochs 1 --ppo-epochs 1 --batch-size 1 --device cuda --policy-backend=gemini > custom_ppo.log 2>&1"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[$(date)] tmux session $SESSION already exists; exiting." >> "$LOG"
  exit 0
fi

echo "[$(date)] waiting for GPU to drop below ${MEM_THRESHOLD}MiB used..." >> "$LOG"

while true; do
  used=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    used=$((used + line))
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

  echo "[$(date)] GPU used ${used}MiB" >> "$LOG"

  if (( used <= MEM_THRESHOLD )); then
    echo "[$(date)] GPU idle enough; launching PPO session." >> "$LOG"
    tmux new-session -d -s "$SESSION" "$COMMAND"
    echo "[$(date)] started tmux session $SESSION." >> "$LOG"
    exit 0
  fi

  sleep "$CHECK_INTERVAL"
  echo "[$(date)] checked; GPU still busy." >> "$LOG"
done
