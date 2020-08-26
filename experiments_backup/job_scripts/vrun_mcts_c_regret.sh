#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_rt=8:0:0
#$ -l h_vmem=4G
# Load ability to use Python
module load python
# Activate virtualenv
source myenv/bin/activate
# Run experiments
python3 run_experiment.py --num_episodes 100 --agent MCTSAgent --agents MCTSAgent --mcts_type 222
deactivate