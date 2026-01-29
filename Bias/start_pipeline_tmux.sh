#!/bin/bash
# Launch smart fix pipeline in tmux session

SESSION_NAME="smart_fix_pipeline"
WORK_DIR="/root/Bias"

# Kill existing session if running
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME -c $WORK_DIR

# Run the pipeline
tmux send-keys -t $SESSION_NAME "cd $WORK_DIR && source /root/.venv/bin/activate && python run_smart_fix_pipeline.py" Enter

# Show session info
echo "✅ Started tmux session: $SESSION_NAME"
echo ""
echo "Monitor progress with:"
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "Or view logs:"
echo "  tail -f $WORK_DIR/smart_fix_step1.log"
echo "  tail -f $WORK_DIR/score_comparison_step2.log"
