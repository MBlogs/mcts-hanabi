# A simple episode runner using the RL environment.

from __future__ import print_function
import sys
import getopt
import rl_env
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from agents.rule_based_agent import RuleBasedAgent

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'RuleBasedAgent':RuleBasedAgent}

class Runner(object):
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = {'players': flags['players']}
    self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
    self.agent_classes = [AGENT_CLASSES[agent_class] for agent_class in flags['agent_classes']]

  def run(self):
    """Run episodes."""
    rewards = []
    for episode in range(flags['num_episodes']):
      observations = self.environment.reset()
      # MB: Allow parsing of different Agents. N
      agents = [agent_class(self.agent_config) for agent_class in self.agent_classes]
      done = False
      episode_reward = 0
      while not done:
        for agent_id, agent in enumerate(agents):
          observation = observations['player_observations'][agent_id]
          action = agent.act(observation)
          if observation['current_player'] == agent_id:
            assert action is not None
            current_player_action = action
          else:
            assert action is None
        # Make an environment step.
        # print('Agent: {} action: {}'.format(observation['current_player'],current_player_action))

        observations, reward, done, unused_info = self.environment.step(current_player_action)
        episode_reward += reward

        # MB: Try a return and DealSpecifc Move upfront for the next player (note this is now in rl_env.Step()
        # print_state(self)
        # return_action = {'action_type': 'RETURN', 'card_index': 0}
        # observations, reward, done, unused_info = self.environment.step(return_action)

      # MB: Rewards seems pretty funky. It's zero for all non-perfect games? A: Yes may want to change that
      rewards.append(episode_reward)
      # print('Running episode: %d' % episode)
      print('Episode {}, Score: {}'.format(episode, self.environment.fireworks_score()))
    return rewards

  def print_state(self):
    self.environment.print_state()

if __name__ == "__main__":
  # MB: agent_class changed to agent_classes
  flags = {'players': 3, 'num_episodes': 400, 'agent_classes': ['RuleBasedAgent', 'RuleBasedAgent', 'RuleBasedAgent']}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class='])
  if arguments:
    sys.exit('usage: rl_env_example.py [options]\n'
             '--players       number of players in the game.\n'
             '--num_episodes  number of game episodes to run.\n'
             '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
  # MB: Added check that the number of agent classes is equal to number of players
  if len(flags['agent_classes']) != flags['players']:
    sys.exit('Number of agent classes not same as number of players')

  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)
  runner = Runner(flags)
  runner.run()
