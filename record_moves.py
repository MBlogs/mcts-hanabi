# Record a game by tracking moves and the game state observations
from pyhanabi import HanabiMoveType
from pyhanabi import HanabiCard
num_rank = [3, 2, 2, 2, 1]

class RecordMoves(object):

  def __init__(self, players):
    self._stat_list = ["score", "moves", "regret"
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
    return {s: 0 for s in self._stat_list}

  def update(self, move, observation, action_player, elapsed_time):
    """Update game stats by passing the action taken and the new state observation."""
    debug = False
    self.game_stats["score"] = self._fireworks_score(observation)
    self.player_stats[action_player]["score"] = self._fireworks_score(observation)
    self._update_stat("elapsed_time", elapsed_time, action_player)
    self._update_stat("moves", 1, action_player)

    # Reveal
    if move.type() == HanabiMoveType.REVEAL_RANK or move.type() == HanabiMoveType.REVEAL_COLOR:
      self._update_stat("information", 1, action_player)
      if move.type() == HanabiMoveType.REVEAL_RANK:
        self._update_stat("information_rank", 1, action_player)
      elif move.type() == HanabiMoveType.REVEAL_COLOR:
        self._update_stat("information_color", 1, action_player)

    # Discard
    if move.type() == HanabiMoveType.DISCARD:
      self._update_stat("discard", 1, action_player)
      card = observation["discard_pile"][-1]
      # Was it a critical discard?
      if self._critical_discard(card, observation):
        self._update_stat("discard_critical", 1, action_player)
        self._update_stat("regret", 5 - card["rank"], action_player)
      # Was it a safe discard?
      elif self._safe_discard(card, observation):
        self._update_stat("discard_safe", 1, action_player)
      else:
        self._update_stat("discard_useful", 1, action_player)

    # Play
    if move.type() == HanabiMoveType.PLAY:
      self._update_stat("play", 1, action_player)
      # If life tokens depleted, it failed.
      if observation["life_tokens"] < self.recorded_observation["life_tokens"]:
        self._update_stat("play_fail", 1, action_player)
        card = observation["discard_pile"][-1]
        # If played card (which gets discarded) was also critical, regret
        if self._critical_discard(card, observation):
          self._update_stat("regret", 5 - card["rank"], action_player)
        # If it actually ended the game, regret missed potential
        if observation["life_tokens"] == 0:
          self._update_stat("regret", self._end_game_regret(observation), action_player)
      else:
        self._update_stat("play_success", 1, action_player)

    self.recorded_observation = observation
    if debug: print(f"record_moves.update: Game {self.game_stats}")
    if debug: print(f"record_moves.update: Players {self.player_stats}")

  def _update_stat(self, stat, increment, action_player):
    self.game_stats[stat] += increment
    self.player_stats[action_player][stat] += increment

  def _end_game_regret(self, observation):
    """Game has ended. How much better could we do?"""
    # Note: Copied from Ruleset
    discarded_cards = {}
    max_fireworks = {'R': 5, 'Y': 5, 'G': 5, 'W': 5, 'B': 5}
    for card in observation['discard_pile']:
      color = card['color']
      rank = card['rank']
      label = str(color) + str(rank)
      if label not in discarded_cards:
        discarded_cards[label] = 1
      else:
        discarded_cards[label] += 1
    for label in discarded_cards:
      color = label[0]
      rank = int(label[1])
      number_in_discard = discarded_cards[label]
      if number_in_discard >= num_rank[rank]:
        if max_fireworks[color] >= rank:
          max_fireworks[color] = rank
    max_fireworks_score = sum(v for k,v in max_fireworks.items())
    fireworks_score = self._fireworks_score(observation)
    return max_fireworks_score - fireworks_score


  def _fireworks_score(self, observation):
    return sum(v for k,v in observation["fireworks"].items())

  def _critical_discard(self, card, observation):
    if self._safe_discard(card, observation):
      return False
    num = self._count_card(card, observation["discard_pile"])
    if num == num_rank[card["rank"]]:
      # Card rank starts at 0, fireworks 1
      if observation["fireworks"][card["color"]] < card["rank"]+1:
        return True
    return False

  def _count_card(self, card, pile):
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
      num = self._count_card(check_card, observation["discard_pile"])
      if num == num_rank[rank]:
        return True
    return False

  def regret(self):
    return self.game_stats['regret']


