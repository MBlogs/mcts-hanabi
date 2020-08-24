#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_rt=1:0:0
#$ -l h_vmem=4G
# Load ability to use Python
module load python
# Activate virtualenv
source myenv/bin/activate
# Run experiments_store
python3 run_experiment.py --players 2 --num_episodes 250 --agents OuterAgent --agent OuterAgent
python3 run_experiment.py --players 3 --num_episodes 250 --agents OuterAgent --agent OuterAgent
python3 run_experiment.py --players 4 --num_episodes 250 --agents OuterAgent --agent OuterAgent
python3 run_experiment.py --players 5 --num_episodes 250 --agents OuterAgent --agent OuterAgent
deactivate