This project implements an Redterminising Information Set Monte Carlo Tree Search agent into the hanabi\_learning\_environment research platform for Hanabi experiments developed by Google Deep Mind available at: https://github.com/deepmind/hanabi-learning-environment
New moves Return and DealSpecific were added to the underlying C++ framework to allow direct state manipulation (swapping cards in and out). Observation encoding and different Hanabi variations are not supported.

### Getting started
Instructions for installing the learning environment in a new linux environment:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python3-pip3   # if you don't already have pip
pip install numpy                   # Python package dependency
pip install cffi                    # Python package dependency
git clone https://github.com/MBlogs/mcts-hanabi
cd mcts-hanabi
cmake .                             # Compile
make                       
```
### Running Experiments
```
python3 run_experiment.py --num_episodes 1 --players 3 --agent HumanAgent --agents MCTSAgent --mcts_types 000
```
### Experiment Parameters
Supported parameters and values include:
```
num_episodes: integer. Number of games to include in experiment.
players: integer. Number of players in the game.
agent: First player will be of this type.
agent_classs: Remaining players will be of this type
mcts_type: string. Types for the MCTS agents, each character corresponding to a player position.
```
Supported Agent Classes are:
- VanDenBerghAgent
- FlawedAgent
- MCTSAgent
- OuterAgent
- InnerAgent
- PiersAgent
- IGGIAgent
- LegalRandomAgent
- MuteAgent
- HumanAgent

Type of MCTS agent in a player position is determined by the corresponding character of the mcts_types string. See agents.mcts.mcts_agent.py for full list of possible types

### Experiment Results
Experiment result prints out Python code that defines a list of Experiments objects.
Raw data from experiment runs can be found in experiments.
experiments/analyse_experiment.ipynb is a notebook that defines the Experiment class to extract this data, and produces the summary tables and graphs seen in the paper