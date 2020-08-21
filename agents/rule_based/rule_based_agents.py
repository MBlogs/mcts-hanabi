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
  """Chooses randomly from set of legal moves"""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)
    self.rules = [Ruleset.legal_random]
    self.rulebased = RulebasedAgent(self.rules)

  def act(self, observation):
    return self.rulebased.get_move(observation)


class VanDenBerghAgent(Agent):
  """High performing Rule Based agent from literature"""

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
                  Ruleset.tell_most_information_factory(False),
                  Ruleset.discard_probably_useless_factory(0)]
    self.rulebased = RulebasedAgent(self.rules)

  def act(self, observation):
    return self.rulebased.get_move(observation)

class FlawedAgent(Agent):
  """Agent that plays card under little information"""

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
    self.rulebased = RulebasedAgent(self.rules)

  def act(self, observation):
    return self.rulebased.get_move(observation)

class IGGIAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)

    # self.rules = [Ruleset.play_safe_card,Ruleset.tell_playable_card_outer,Ruleset.discard_randomly,Ruleset.legal_random]
    self.rules = [Ruleset.play_if_certain,
                  Ruleset.play_safe_card,
                  Ruleset.tell_playable_card_outer,
                  Ruleset.osawa_discard,
                  Ruleset.discard_oldest_first,
                  Ruleset.legal_random]

    self.rulebased = RulebasedAgent(self.rules)

  def act(self, observation):
    return self.rulebased.get_move(observation)


class PiersAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)

    # self.rules = [Ruleset.play_safe_card,Ruleset.tell_playable_card_outer,Ruleset.discard_randomly,Ruleset.legal_random]
    self.rules = [Ruleset.hail_mary,
                  Ruleset.play_safe_card,
                  Ruleset.play_probably_safe_factory(0.6,True),
                  Ruleset.tell_anyone_useful_card,
                  Ruleset.tell_dispensable_factory(3),
                  Ruleset.osawa_discard,
                  Ruleset.discard_oldest_first,
                  Ruleset.tell_randomly,
                  Ruleset.discard_randomly]

    self.rulebased = RulebasedAgent(self.rules)

  def act(self, observation):
    return self.rulebased.get_move(observation)

  class OuterAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, config, *args, **kwargs):
      """Initialize the agent."""
      self.config = config
      # Extract max info tokens or set default to 8.
      self.max_information_tokens = config.get('information_tokens', 8)

      # self.rules = [Ruleset.play_safe_card,Ruleset.tell_playable_card_outer,Ruleset.discard_randomly,Ruleset.legal_random]
      self.rules = [Ruleset.play_safe_card,
                    Ruleset.osawa_discard,
                    Ruleset.tell_playable_card_outer,
                    Ruleset.tell_unknown,
                    Ruleset.discard_randomly]

      self.rulebased = RulebasedAgent(self.rules)

    def act(self, observation):
      return self.rulebased.get_move(observation)

  class OuterAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, config, *args, **kwargs):
      """Initialize the agent."""
      self.config = config
      # Extract max info tokens or set default to 8.
      self.max_information_tokens = config.get('information_tokens', 8)

      # self.rules = [Ruleset.play_safe_card,Ruleset.tell_playable_card_outer,Ruleset.discard_randomly,Ruleset.legal_random]
      self.rules = [Ruleset.play_safe_card,
                    Ruleset.osawa_discard,
                    Ruleset.tell_playable_card_outer,
                    Ruleset.tell_unknown,
                    Ruleset.discard_randomly]

      self.rulebased = RulebasedAgent(self.rules)

    def act(self, observation):
      return self.rulebased.get_move(observation)