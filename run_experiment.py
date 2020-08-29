# A simple episode runner using the RL environment.

from __future__ import print_function
import sys
import getopt
from rl_env import make
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.rule_based.rule_based_agents import OuterAgent
from agents.rule_based.rule_based_agents import InnerAgent
from agents.rule_based.rule_based_agents import PiersAgent
from agents.rule_based.rule_based_agents import IGGIAgent
from agents.rule_based.rule_based_agents import LegalRandomAgent
from agents.rule_based.rule_based_agents import FlawedAgent
from agents.rule_based.rule_based_agents import MuteAgent
from agents.mcts.mcts_agent import MCTSAgent

AGENT_CLASSES = {'VanDenBerghAgent': VanDenBerghAgent,'FlawedAgent':FlawedAgent, 'MCTSAgent': MCTSAgent
                  , 'OuterAgent':OuterAgent, 'InnerAgent':InnerAgent, 'PiersAgent':PiersAgent, 'IGGIAgent':IGGIAgent
                  , 'LegalRandomAgent':LegalRandomAgent, 'MuteAgent':MuteAgent}

class Runner(object):
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = {'players': flags['players'], 'player_id':0, 'mcts_types':flags['mcts_types']}
    self.environment = make('Hanabi-Full', num_players=flags['players'])
    self.agent_classes = [AGENT_CLASSES[agent_class] for agent_class in flags['agent_classes']]


  def run(self):
    """Run episodes."""
    game_stats = []
    player_stats = []
    agents = []

    # MB: Pass absolute player_id upfront to all agents (MCTS needs this for forward model)

    for i in range(len(self.agent_classes)):
      self.agent_config.update({'player_id': i}) #change player_id
      agents.append(self.agent_classes[i](self.agent_config))
      player_stats.append([])

    print("]") # end mcts_config
    print(",progress=[", end="")
    errors = 0

    for episode in range(flags['num_episodes']):
      done = False
      observations = self.environment.reset()
      try:
        while not done:
          for agent_id, agent in enumerate(agents):
            observation = observations['player_observations'][agent_id]
            # MB: MCTSAgent needs to be passed full state to act as base for MCTS
            # MB: Note that it replaces it's hand before each rollout so not 'cheating' by knowing the full state
            if isinstance(agent, MCTSAgent):
              action = agent.act(observation, self.environment.state)
            else:
              action = agent.act(observation)
            if observation['current_player'] == agent_id:
              assert action is not None
              current_player_action = action
            else:
              assert action is None
          observations, reward, done, unused_info = self.environment.step(current_player_action)
        print(self.environment.progress(), end=",")
        game_stats.append(self.environment.game_stats())
        for i in range(len(self.agent_classes)):
          player_stats[i].append(self.environment.player_stats(i))
      except Exception as e:
        raise e
        errors += 1

    print("]")
    print(f",scores = {[g['score'] for g in game_stats]}")
    print(f",stats_keys={list(game_stats[0].keys())}")
    print(f",game_stats = {self.simplify_stats(game_stats)}")
    print(f",player_stats = {[self.simplify_stats(p) for p in player_stats]}")
    avg_progress = sum([g["progress"] for g in game_stats]) / flags['num_episodes']
    avg_score = sum([g["score"] for g in game_stats]) / flags['num_episodes']
    avg_time = sum([p["elapsed_time"]/max(p["moves"], 1) for p in player_stats[0]]) / flags['num_episodes']
    print(f",avg_progress={avg_progress}")
    print(f",avg_score={avg_score}")
    print(f",avg_time={avg_time}")
    print(f",errors={errors}")
    print("),")

  def simplify_stats(self, stats):
    """Extract just the numbers from the stats"""
    return [list(g.values()) for g in stats]

  def print_state(self):
    self.environment.print_state()

if __name__ == "__main__":
  # MB: agent: Player of interest. agent: fill in remaining spaces
  flags = {'players': 3, 'num_episodes': 1
    ,'agent':'VanDenBerghAgent', 'agents':'VanDenBerghAgent'
    , 'mcts_types': '000'}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent=',
                                      'agents=',
                                      'mcts_types='])
  if arguments:
    sys.exit('usage: rl_env_example.py [options]\n'
             '--players       number of players in the game.\n'
             '--num_episodes  number of game episodes to run.\n'
             '--agent  class name of agent of interest {}\n'
             '--agents  class name of agent to play off: {}\n'
             '--mcts_types 00000 each character is the type of the mcts agent in that position, see mcts_agent._edit_mcts_config'
             ''.format(' or '.join(AGENT_CLASSES.keys())))

  # Convert any extra options into the flags
  for flag, value in options:
    flag = flag[2:]  # Strip leading --
    flags[flag] = type(flags[flag])(value)

  # agent_classes lists the players of the game
  flags['agent_classes'] = [flags['agent']] + [flags['agents'] for _ in range(1, flags["players"])]

  # Agents list needs to be same size as number of players declared
  if len(flags['agent_classes']) != flags['players']:
    sys.exit(f'Number of agent classes:{len(flags["agent_classes"])} not same as number of players: {flags["players"]}')

  #Print the config
  print("experiments = [Experiment(")
  print(f"flags = {flags}")
  print(",mcts_configs = [")
  runner = Runner(flags)
  runner.run()
  print("]")
