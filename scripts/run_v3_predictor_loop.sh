#!/bin/bash
# ==============================================================================
#  [V3 Predictor Daemon]
#  Run the Pattern-Based Direction Classification loop periodically.
#  This ensures `trading_bot.py` can fetch the latest geometric predictions
#  instantly without blocking its main logic loop.
# ==============================================================================

# 1. Initialize Conda Environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quant_ai

# 2. Path Setup
PROJ_DIR="/home/llewyn/crypto-scalping"
SCRIPT="$PROJ_DIR/quant/live_30m_direction_quant.py"
OUT_FILE="$PROJ_DIR/analysis/direction_v3_latest.txt"

echo "[V3 Predictor Daemon] Starting..."
echo "[V3 Predictor Daemon] Logging output to: $OUT_FILE"

# 3. Infinite Loop (Run & Sleep)
while true; do
    START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[V3 Predictor Daemon] [$START_TIME] Running prediction iteration..."

    # Run the script and write output
    python3 "$SCRIPT" > "$OUT_FILE.tmp" 2>&1

    # Atomic move to prevent read corruption during write
    mv "$OUT_FILE.tmp" "$OUT_FILE"

    END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[V3 Predictor Daemon] [$END_TIME] Finished. Sleeping for 45 seconds..."

    sleep 45
done
