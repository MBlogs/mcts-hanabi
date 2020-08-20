# A simple episode runner using the RL environment.

from __future__ import print_function
import sys
import getopt
import rl_env
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.rule_based.rule_based_agents import FlawedAgent
from agents.mcts.mcts_agent import MCTSAgent

AGENT_CLASSES = {'VanDenBerghAgent': VanDenBerghAgent,'FlawedAgent':FlawedAgent, 'MCTSAgent': MCTSAgent}

class Runner(object):
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = {'players': flags['players'], 'player_id':0} #player_id changes per Agent
    self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
    self.agent_classes = [AGENT_CLASSES[agent_class] for agent_class in flags['agent_classes']]


  def run(self):
    """Run episodes."""
    game_stats = []
    player_stats = [[],[],[]]

    for episode in range(flags['num_episodes']):
      observations = self.environment.reset()

      # MB: Pass absolute player_id upfront to all agents (MCTS needs this for forward model)
      agents = []
      for i in range(len(self.agent_classes)):
        self.agent_config.update({'player_id': i})
        agents.append(self.agent_classes[i](self.agent_config))

      done = False
      episode_reward = 0
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

      print('Episode {}, Score: {}'.format(episode, self.environment.fireworks_score()))
      game_stats.append(self.environment.record_moves.game_stats)
      for i in range(len(self.agent_classes)):
        player_stats[i].append(self.environment.record_moves.player_stats[i])

    print(f"GameStats: {game_stats}")
    print(f"PlayerStats: {player_stats}")
    avg_score = sum([g["score"] for g in game_stats]) / flags['num_episodes']
    avg_time = sum([p["elapsed_time"]/p["moves"] for p in player_stats[0]]) / flags['num_episodes']
    print(f"Average Score: {avg_score}")
    print(f"Average Think Time: {avg_time}")

  def print_state(self):
    self.environment.print_state()

if __name__ == "__main__":
  # MB: agent_class changed to agent_classes
  flags = {'players': 3, 'num_episodes': 1, 'agent_classes': ['MCTSAgent', 'MCTSAgent', 'MCTSAgent']}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class='])
  if arguments:
    sys.exit('usage: rl_env_example.py [options]\n'
             '--players       number of players in the game.\n'
             '--num_episodes  number of game episodes to run.\n'
             '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))

  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)

  # Agents list needs to be same size as number of players declared
  if len(flags['agent_classes']) != flags['players']:
    sys.exit('Number of agent classes not same as number of players')

  runner = Runner(flags)
  runner.run()
