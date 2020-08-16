from rl_env import Agent
from agents.rule_based.ruleset import Ruleset

# Agents sourced at https://github.com/rocanaan

class RulebasedAgent():
  """Class of agents that follow rules"""

  def __init__(self, rules):
    self.rules = rules
    self.totalCalls = 0
    self.histogram = [0 for i in range(len(rules) + 1)]

  def get_move(self, observation):
    if observation['current_player_offset'] == 0:
      for index, rule in enumerate(self.rules):
        action = rule(observation)
        if action is not None:
          # print(rule)
          self.histogram[index] += 1
          self.totalCalls += 1
          return action
      self.histogram[-1] += 1
      self.totalCalls += 1
      return Ruleset.legal_random(observation)
    return None

  def print_histogram(self):
    if self.totalCalls > 0:
      print([calls / self.totalCalls for calls in self.histogram])


class LegalRandomAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)
    self.rules = [Ruleset.legal_random]
    print(self.rules)
    self.rulebased = RulebasedAgent(self.rules)

  def act(self, observation):
    return self.rulebased.get_move(observation)


class VanDenBerghAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)
    self.rules = [Ruleset.play_probably_safe_factory(0.6, True),
                  Ruleset.play_safe_card,
                  Ruleset.discard_probably_useless_factory(0.99),
                  Ruleset.tell_anyone_useful_card,
                  Ruleset.tell_anyone_useless_card,
                  Ruleset.tell_most_information,
                  Ruleset.discard_probably_useless_factory(0)]

    self.rulebased = RulebasedAgent(self.rules)

  def act(self, observation):
    return self.rulebased.get_move(observation)

class FlawedAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)
    self.rules = [Ruleset.play_safe_card,
                  Ruleset.play_probably_safe_factory(0.25),
                  Ruleset.tell_randomly,
                  Ruleset.osawa_discard,
                  Ruleset.discard_oldest_first,
                  Ruleset.discard_randomly]

    print(self.rules)

    self.rulebased = RulebasedAgent(self.rules)

  def act(self, observation):
    return self.rulebased.get_move(observation)
