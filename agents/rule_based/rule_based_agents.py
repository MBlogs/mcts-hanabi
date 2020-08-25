from rl_env import Agent
from agents.rule_based.ruleset import Ruleset
# Agents originally sourced at https://github.com/rocanaan

class RulebasedAgent(Agent):
  """Class of agents that follow rules"""
  def __init__(self,config,rules):
    self.rules = rules
    self.max_information_tokens = config.get('information_tokens', 8)
    self.totalCalls = 0
    self.histogram = [0 for i in range(len(rules) + 1)]

  def get_move(self, observation):
    debug = True
    if observation['current_player_offset'] == 0:
      for index, rule in enumerate(self.rules):
        action = rule(observation)
        if action is not None:
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

  def act(self, observation):
    return self.get_move(observation)


class LegalRandomAgent(RulebasedAgent):
  """Chooses randomly from set of legal moves"""
  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    rules = [Ruleset.legal_random]
    super().__init__(config, rules)


class VanDenBerghAgent(RulebasedAgent):
  """High performing Rule Based agent from literature"""
  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    rules = [Ruleset.play_probably_safe_factory(0.6, True),
                  Ruleset.play_safe_card,
                  Ruleset.discard_probably_useless_factory(0.99),
                  Ruleset.tell_anyone_useful_card,
                  Ruleset.tell_anyone_useless_card,
                  Ruleset.tell_most_information_factory(False),
                  Ruleset.discard_probably_useless_factory(0)]
    super().__init__(config, rules)


class FlawedAgent(RulebasedAgent):
  """Agent that plays card under little information"""
  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    rules = [Ruleset.play_safe_card,
                  Ruleset.play_probably_safe_factory(0.25),
                  Ruleset.tell_randomly,
                  Ruleset.osawa_discard,
                  Ruleset.discard_oldest_first,
                  Ruleset.discard_randomly]
    super().__init__(config, rules)


class IGGIAgent(RulebasedAgent):
  """Agent that applies a simple heuristic."""
  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    rules = [Ruleset.play_if_certain,
                  Ruleset.play_safe_card,
                  Ruleset.tell_playable_card_outer,
                  Ruleset.osawa_discard,
                  Ruleset.discard_oldest_first,
                  Ruleset.legal_random]
    super().__init__(config, rules)


class PiersAgent(RulebasedAgent):
  """Agent that applies a simple heuristic."""
  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    rules = [Ruleset.hail_mary,
                  Ruleset.play_safe_card,
                  Ruleset.play_probably_safe_factory(0.6,True),
                  Ruleset.tell_anyone_useful_card,
                  Ruleset.tell_dispensable_factory(3),
                  Ruleset.osawa_discard,
                  Ruleset.discard_oldest_first,
                  Ruleset.tell_randomly,
                  Ruleset.discard_randomly]
    super().__init__(config, rules)


class OuterAgent(RulebasedAgent):
  """Agent that applies a simple heuristic."""
  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    rules = [Ruleset.play_safe_card,
                  Ruleset.osawa_discard,
                  Ruleset.tell_playable_card_outer,
                  Ruleset.tell_unknown,
                  Ruleset.discard_randomly]
    super().__init__(config, rules)


class InnerAgent(RulebasedAgent):
  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    rules = [Ruleset.play_safe_card
             , Ruleset.osawa_discard
             , Ruleset.tell_playable_card
             , Ruleset.tell_randomly
             , Ruleset.discard_randomly]
    super().__init__(config, rules)

class MuteAgent(RulebasedAgent):
  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    rules = [Ruleset.play_probably_safe_factory(0.6, True),
                  Ruleset.play_safe_card,
                  Ruleset.discard_probably_useless_factory(0.99),
                  Ruleset.discard_probably_useless_factory(0),
                  Ruleset.play_probably_safe_factory(0, False)]
    super().__init__(config, rules)
