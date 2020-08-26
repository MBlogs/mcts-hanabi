#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_rt=15:0:0
#$ -l h_vmem=4G
# Load ability to use Python
module load python
# Activate virtualenv
source myenv/bin/activate
# Run experiments_store
python3 run_experiment.py --num_episodes 100 --agent MuteAgent --agents MCTSAgent --mcts_type ddd
python3 run_experiment.py --num_episodes 100 --agent MuteAgent --agents MCTSAgent --mcts_type iii
python3 run_experiment.py --num_episodes 100 --agent MuteAgent --agents MCTSAgent --mcts_type kkk
deactivate
