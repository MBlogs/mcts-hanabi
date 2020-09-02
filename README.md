This project implements an MCTS agent into the hanabi\_learning\_environment research platform for Hanabi experiments developed by Google Deep Mind available at: https://github.com/deepmind/hanabi-learning-environment

### Getting started
Install the learning environment from a new linux environment:
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
python3 run_experiment.py --num_players 3 --agent HumanAgent --agents MCTSAgent
```
### Experiment Parameters
Supported parameters and values include:
```
num_players: integer. Number of agents in the game.
agent: First player will be of this type.
agent_classs: Remaining players will be of this type
mcts_type: Type of MCTS agent.
```
Supported Agent Classes are:
- VanDenBerghAgent
- FlawedAgent,
- MCTSAgent
- OuterAgent
- InnerAgent
- PiersAgent
- IGGIAgent
- LegalRandomAgent
- MuteAgent
- HumanAgent

### Experiment Results
Raw data from experiment runs can be found in experiments folder.
analyse_experiment.ipynb is a notebook that extracts this data and produces the summary tables and graphs seen in the paper
