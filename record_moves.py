# Record a game by tracking moves and the game state observations
from pyhanabi import HanabiMoveType
from pyhanabi import HanabiCard
num_rank = [3, 2, 2, 2, 1]

class RecordMoves(object):

  def __init__(self, players):
    self._stat_list = ["score", "moves"
      , "discard", "discard_critical", "discard_useful", "discard_safe"
      , "play", "play_success", "play_fail"
      , "information", "information_color", "information_rank", "elapsed_time"]
    self.recorded_observation = None
    self.players = players
    self.game_stats = self.default_stats()
    self.player_stats = [self.default_stats() for _ in range(self.players)]


  def reset(self, observations):
    self.recorded_observation = observations
    self.game_stats = self.default_stats()
    self.player_stats = [self.default_stats() for _ in range(self.players)]

  def default_stats(self):
    return {s:0 for s in self._stat_list}

  def update(self, move, observation, action_player, elapsed_time):
    """Update game stats by passing the action taken and the new state observation."""
    debug = False
    self.game_stats["elapsed_time"] += elapsed_time
    self.player_stats[action_player]["elapsed_time"] += elapsed_time
    self.game_stats["score"] = self._fireworks_score(observation)
    self.player_stats[action_player]["score"] = self._fireworks_score(observation)
    self.game_stats["moves"] += 1
    self.player_stats[action_player]["moves"] += 1

    # Reveal
    if move.type() == HanabiMoveType.REVEAL_RANK or move.type() == HanabiMoveType.REVEAL_COLOR:
      self.game_stats["information"] += 1
      self.player_stats[action_player]["information"] += 1
      if move.type() == HanabiMoveType.REVEAL_RANK:
        self.game_stats["information_rank"] += 1
        self.player_stats[action_player]["information_rank"] += 1
      elif move.type() == HanabiMoveType.REVEAL_COLOR:
          self.game_stats["information_color"] += 1
          self.player_stats[action_player]["information_color"] += 1

    # Discard
    if move.type() == HanabiMoveType.DISCARD:
      self.game_stats["discard"] += 1
      self.player_stats[action_player]["discard"] += 1
      # Was it a critical discard?
      card = observation["discard_pile"][-1]
      if self._critical_discard(card, observation):
        self.game_stats["discard_critical"] += 1
        self.player_stats[action_player]["discard_critical"] += 1
      # Was it a safe discard?
      elif self._safe_discard(card, observation):
        self.game_stats["discard_safe"] += 1
        self.player_stats[action_player]["discard_safe"] += 1
      else:
        self.game_stats["discard_useful"] += 1
        self.player_stats[action_player]["discard_useful"] += 1

    # Play
    if move.type() == HanabiMoveType.PLAY:
      self.game_stats["play"] += 1
      self.player_stats[action_player]["play"] += 1
      # If life tokens depleted, it failed
      if observation["life_tokens"] < self.recorded_observation["life_tokens"]:
        self.game_stats["play_fail"] += 1
        self.player_stats[action_player]["play_fail"] += 1
      else:
        self.game_stats["play_success"] += 1
        self.player_stats[action_player]["play_success"] += 1

    self.recorded_observation = observation
    if debug: print(f"record_moves.update: Game {self.game_stats}")
    if debug: print(f"record_moves.update: Players {self.player_stats}")

  def _max_potential_score(self, observation):
    max_potential = {c:5 for c,v in observation["fireworks"].items()}


  def _fireworks_score(self,observation):
    return sum(v for k,v in observation["fireworks"].items())

  def _critical_discard(self, card, observation):
    if self._safe_discard(card, observation):
      return False
    num = self.count_card(card, observation["discard_pile"])
    if num == num_rank[card["rank"]]:
      # Card rank starts at 0, fireworks 1
      if observation["fireworks"][card["color"]] < card["rank"]+1:
        return True
    return False

  def count_card(self, card, pile):
    num = 0
    for discarded_card in pile:
      if discarded_card["rank"] == card["rank"] and discarded_card["color"] == card["color"]:
        num += 1
    return num

  def _safe_discard(self, card, observation):
    # If firework is already passed this rank
    if observation["fireworks"][card["color"]] >= card["rank"]+1:
      return True
    # If a firework is already cut off
    for rank in range(observation["fireworks"][card["color"]], card["rank"]):
      check_card = {'color': card["color"], "rank": rank}
      num = self.count_card(check_card, observation["discard_pile"])
      if num == num_rank[rank]:
        return True
    return False

  def critical_discards(self):
    return self.game_stats["discard_critical"]

  def play_fail(self):
    return self.game_stats["play_fail"]


