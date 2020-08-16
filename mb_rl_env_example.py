# A simple episode runner using the RL environment.

from __future__ import print_function
import sys
import getopt
import rl_env
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.mcts.mcts_agent import MCTSAgent

AGENT_CLASSES = {'VanDenBerghAgent': VanDenBerghAgent, 'MCTSAgent': MCTSAgent}

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
    rewards = []

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
          # MB: Note that it replaces it's hand with random before each rollout so not 'cheating' by knowing the full state
          if isinstance(agent, MCTSAgent):
            action = agent.act(observation, self.environment.state)
          else:
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
  flags = {'players': 3, 'num_episodes': 5, 'agent_classes': ['MCTSAgent', 'MCTSAgent', 'MCTSAgent']}
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

    # MB: Added check that the number of agent classes is equal to number of players
  if len(flags['agent_classes']) != flags['players']:
    sys.exit('Number of agent classes not same as number of players')

  runner = Runner(flags)
  runner.run()
