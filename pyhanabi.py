# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python interface to Hanabi code."""
import os
import re
import cffi
import enum
import sys
import random

DEFAULT_CDEF_PREFIXES = (None, ".", os.path.dirname(__file__), "/include")
DEFAULT_LIB_PREFIXES = (None, ".", os.path.dirname(__file__), "/lib")
PYHANABI_HEADER = "pyhanabi.h"
PYHANABI_LIB = ["libpyhanabi.so", "libpyhanabi.dylib"]
COLOR_CHAR = ["R", "Y", "G", "W", "B"]  # consistent with hanabi_lib/util.cc
CHANCE_PLAYER_ID = -1

ffi = cffi.FFI()
lib = None
cdef_loaded_flag = False
lib_loaded_flag = False


if sys.version_info < (3,):
  def encode_ffi_string(x):
    return ffi.string(x)
else:
  def encode_ffi_string(x):
    return str(ffi.string(x), 'ascii')

def try_cdef(header=PYHANABI_HEADER, prefixes=DEFAULT_CDEF_PREFIXES):
  """Try parsing library header file. Must be called before any pyhanabi calls.

  Args:
    header: filename of pyhanabi header file.
    prefixes: list of paths to search for pyhanabi header file.
  Returns:
    True if header was successfully parsed, False on failure.
  """
  global cdef_loaded_flag
  if cdef_loaded_flag: return True
  for prefix in prefixes:
    try:
      cdef_file = header if prefix is None else prefix + "/" + header
      lines = open(cdef_file).readlines()
      reading_cdef = False
      cdef_string = ""
      for line in lines:
        line = line.rstrip()
        if re.match("extern *\"C\" *{", line):
          reading_cdef = True
          continue
        elif re.match("} */[*] *extern *\"C\" *[*]/", line):
          reading_cdef = False
          continue
        if reading_cdef:
          cdef_string = cdef_string + line + "\n"
      ffi.cdef(cdef_string)
      cdef_loaded_flag = True
      return True
    except IOError:
      pass
  return False


def try_load(library=None, prefixes=DEFAULT_LIB_PREFIXES):
  """Try loading library. Must be called before any pyhanabi calls.

  Args:
    library: filename of pyhanabi library file.
    prefixes: list of paths to search for pyhanabi library file.
  Returns:
    True if library was successfully loaded, False on failure.
  """
  global lib_loaded_flag
  global lib
  if lib_loaded_flag: return True
  if library is None:
    libnames = PYHANABI_LIB
  elif type(library) in (list, tuple):
    libnames = library
  else:
    libnames = (library,)
  for prefix in prefixes:
    for libname in libnames:
      try:
        lib_file = libname if prefix is None else prefix + "/" + libname
        lib = ffi.dlopen(lib_file)
        lib_loaded_flag = True
        return True
      except OSError:
        pass
  return False


def cdef_loaded():
  """Return True if pyhanabi header has been successfully parsed."""
  return cdef_loaded_flag


def lib_loaded():
  """Return True if pyhanabi library has been successfully loaded."""
  return lib_loaded_flag


def color_idx_to_char(color_idx):
  """Helper function for converting color index to a character.

  Args:
    color_idx: int, index into color char vector.

  Returns:
    color_char: str, A single character representing a color.

  Raises:
    AssertionError: If index is not in range.
  """
  assert isinstance(color_idx, int)
  if color_idx == -1:
    return None
  else:
    return COLOR_CHAR[color_idx]


def color_char_to_idx(color_char):
  r"""Helper function for converting color character to index.

  Args:
    color_char: str, Character representing a color.

  Returns:
    color_idx: int, Index into a color array \in [0, num_colors -1]

  Raises:
    ValueError: If color_char is not a valid color.
  """
  assert isinstance(color_char, str)
  try:
    return next(idx for (idx, c) in enumerate(COLOR_CHAR) if c == color_char)
  except StopIteration:
    raise ValueError("Invalid color: {}. Should be one of {}.".format(
        color_char, COLOR_CHAR))


class HanabiCard(object):
  """Hanabi card, with a color and a rank.

  Python implementation of C++ HanabiCard class.
  """

  def __init__(self, color, rank):
    """A simple HanabiCard object.

    Args:
      color: an integer, starting at 0. Colors are in this order RYGWB.
      rank: an integer, starting at 0 (representing a 1 card). In the standard
          game, the largest value is 4 (representing a 5 card).
    """
    self._color = color
    self._rank = rank

  def color(self):
    return self._color

  def rank(self):
    return self._rank

  def __str__(self):
    if self.valid():
      return COLOR_CHAR[self._color] + str(self._rank + 1)
    else:
      return "XX"

  def __repr__(self):
    return self.__str__()

  def __eq__(self, other):
    return self._color == other.color() and self._rank == other.rank()

  def valid(self):
    return self._color >= 0 and self._rank >= 0

  def to_dict(self):
    """Serialize to dict.

    Returns:
      d: dict, containing color and rank of card.
    """
    return {"color": color_idx_to_char(self.color()), "rank": self.rank()}


class HanabiCardKnowledge(object):
  """Accumulated knowledge about color and rank of an initially unknown card.

  Stores two types of knowledge: direct hints about a card, and indirect
  knowledge from hints about other cards.

  For example, say we had two cards that we know nothing about, but our
  partners know are a R1 and B2. Before any hints, both color() and rank()
  return None, and color_plausible(c) and rank_plausible(r) returns True for
  all colors c and ranks r, for both cards.

  Say our partner reveals that our first card is a 1 -- rank index 0. Now for
  the first card we have rank()=0, rank_plausible(0)=True, and
  rank_plausible(r)=False for r != 0. The same hint also tells us the second
  card is NOT a 1 (rank index 0). For the second card, we have rank()=None,
  rank_plausible(0)=False, and rank_plausible(r)=True for r!= 0.

  Note that color() and rank() only consider directly revealed information.
  Both methods will always return None unless the color or rank, respectively,
  are directly revealed. That is, even we have seen hints for all ranks except
  rank index 0, so that rank_plausible(0)=True and rank_plausible(r)=False
  for all r != 0, rank() will still be None.

  Python wrapper of C++ HanabiHand::CardKnowledge class.
  """

  def __init__(self, knowledge):
    self._knowledge = knowledge

  def color(self):
    """Returns color index if exact color was revealed, or None otherwise.

    Does not perform inference to deduce the color from other color hints.
    """
    if lib.ColorWasHinted(self._knowledge):
      return lib.KnownColor(self._knowledge)
    else:
      return None

  def color_plausible(self, color_index):
    """Returns true if we have no hint saying card is not the given color.

    Args:
      color_index: 0-based color index.
    """
    return lib.ColorIsPlausible(self._knowledge, color_index)

  def rank(self):
    """Returns rank index if exact rank was revealed, or None otherwise.

    Does not perform inference to deduce the rank from other rank hints.
    """
    if lib.RankWasHinted(self._knowledge):
      return lib.KnownRank(self._knowledge)
    else:
      return None

  def rank_plausible(self, rank_index):
    """Returns true if we have no hint saying card is not the given rank.

    Args:
      rank_index: 0-based rank index.
    """
    return lib.RankIsPlausible(self._knowledge, rank_index)

  def __str__(self):
    c_string = lib.CardKnowledgeToString(self._knowledge)
    string = encode_ffi_string(c_string)
    lib.DeleteString(c_string)
    return string

  def __repr__(self):
    return self.__str__()

  def to_dict(self):
    """Serialize to dict.

    Returns:
      d: dict, containing color and rank of hint.
    """
    return {"color": color_idx_to_char(self.color()), "rank": self.rank()}


class HanabiMoveType(enum.IntEnum):
  """Move types, consistent with hanabi_lib/hanabi_move.h."""
  INVALID = 0
  PLAY = 1
  DISCARD = 2
  REVEAL_COLOR = 3
  REVEAL_RANK = 4
  DEAL = 5
  RETURN = 6
  DEAL_SPECIFIC = 7

class HanabiMove(object):
  """Description of an agent move or chance event.

  Python wrapper of C++ HanabiMove class.
  """

  def __init__(self, move):
    assert move is not None
    self._move = move

  @property
  def c_move(self):
    return self._move

  def type(self):
    return HanabiMoveType(lib.MoveType(self._move))

  def card_index(self):
    """Returns 0-based card index for PLAY and DISCARD moves."""
    return lib.CardIndex(self._move)

  def target_offset(self):
    """Returns target player offset for REVEAL_XYZ moves."""
    return lib.TargetOffset(self._move)

  def color(self):
    """Returns 0-based color index for REVEAL_COLOR and DEAL moves."""
    return lib.MoveColor(self._move)

  def rank(self):
    """Returns 0-based rank index for REVEAL_RANK and DEAL moves."""
    return lib.MoveRank(self._move)

  @staticmethod
  def get_discard_move(card_index):
    c_move = ffi.new("pyhanabi_move_t*")
    assert lib.GetDiscardMove(card_index, c_move)
    return HanabiMove(c_move)

  @staticmethod
  def get_return_move(card_index):
    c_move = ffi.new("pyhanabi_move_t*")
    assert lib.GetReturnMove(card_index, c_move)
    return HanabiMove(c_move)

  @staticmethod
  def get_deal_specific_move(card_index, player, color, rank):
    c_move = ffi.new("pyhanabi_move_t*")
    assert lib.GetDealSpecificMove(card_index, player, color, rank, c_move)
    return HanabiMove(c_move)

  @staticmethod
  def get_play_move(card_index):
    c_move = ffi.new("pyhanabi_move_t*")
    assert lib.GetPlayMove(card_index, c_move)
    return HanabiMove(c_move)

  @staticmethod
  def get_reveal_color_move(target_offset, color):
    """current player is 0, next player clockwise is target_offset 1, etc."""
    c_move = ffi.new("pyhanabi_move_t*")
    assert lib.GetRevealColorMove(target_offset, color, c_move)
    return HanabiMove(c_move)

  @staticmethod
  def get_reveal_rank_move(target_offset, rank):
    """current player is 0, next player clockwise is target_offset 1, etc."""
    c_move = ffi.new("pyhanabi_move_t*")
    assert lib.GetRevealRankMove(target_offset, rank, c_move)
    return HanabiMove(c_move)

  def __str__(self):
    c_string = lib.MoveToString(self._move)
    string = encode_ffi_string(c_string)
    lib.DeleteString(c_string)
    return string

  def __repr__(self):
    return self.__str__()

  def __del__(self):
    if self._move is not None:
      lib.DeleteMove(self._move)
      self._move = None
    del self

  def to_dict(self):
    """Serialize to dict.

    Returns:
      d: dict, Containing type and information of a hanabi move.

    Raises:
      ValueError: If move type is not supported.
    """
    move_dict = {}
    move_type = self.type()
    move_dict["action_type"] = move_type.name
    if move_type == HanabiMoveType.PLAY or move_type == HanabiMoveType.DISCARD:
      move_dict["card_index"] = self.card_index()
    elif move_type == HanabiMoveType.REVEAL_COLOR:
      move_dict["target_offset"] = self.target_offset()
      move_dict["color"] = color_idx_to_char(self.color())
    elif move_type == HanabiMoveType.REVEAL_RANK:
      move_dict["target_offset"] = self.target_offset()
      move_dict["rank"] = self.rank()
    elif move_type == HanabiMoveType.DEAL:
      move_dict["color"] = color_idx_to_char(self.color())
      move_dict["rank"] = self.rank()
    else:
      raise ValueError("Unsupported move: {}".format(self))

    return move_dict


class HanabiHistoryItem(object):
  """A move that has been made within a game, along with the side-effects.

  For example, a play move simply selects a card index between 0-5, but after
  making the move, there is an associated color and rank for the selected card,
  a possibility that the card was successfully added to the fireworks, and an
  information token added if the firework stack was completed.

  Python wrapper of C++ HanabiHistoryItem class.
  """

  def __init__(self, item):
    self._item = item

  def move(self):
    c_move = ffi.new("pyhanabi_move_t*")
    lib.HistoryItemMove(self._item, c_move)
    return HanabiMove(c_move)

  def player(self):
    return lib.HistoryItemPlayer(self._item)

  def scored(self):
    """Play move succeeded in placing card on fireworks."""
    return bool(lib.HistoryItemScored(self._item))

  def information_token(self):
    """Play/Discard move increased the number of information tokens."""
    return bool(lib.HistoryItemInformationToken(self._item))

  def color(self):
    """Color index of card that was Played/Discarded."""
    return lib.HistoryItemColor(self._item)

  def rank(self):
    """Rank index of card that was Played/Discarded."""
    return lib.HistoryItemRank(self._item)

  def card_info_revealed(self):
    """Returns information about whether color/rank was revealed.

    Indices where card i color/rank matches the reveal move. E.g.,
    for Reveal player 1 color red when player 1 has R1 W1 R2 R4 __ the
    result would be [0, 2, 3].
    """
    revealed = []
    bitmask = lib.HistoryItemRevealBitmask(self._item)
    for i in range(8):  # 8 bits in reveal_bitmask
      if bitmask & (1 << i):
        revealed.append(i)
    return revealed

  def card_info_newly_revealed(self):
    """Returns information about whether color/rank was newly revealed.

    Indices where card i color/rank was not previously known. E.g.,
    for Reveal player 1 color red when player 1 has R1 W1 R2 R4 __ the
    result might be [2, 3].  Cards 2 and 3 were revealed to be red,
    but card 0 was previously known to be red, so nothing new was
    revealed. Card 4 is missing, so nothing was revealed about it.
    """
    revealed = []
    bitmask = lib.HistoryItemNewlyRevealedBitmask(self._item)
    for i in range(8):  # 8 bits in reveal_bitmask
      if bitmask & (1 << i):
        revealed.append(i)
    return revealed

  def deal_to_player(self):
    """player that card was dealt to for Deal moves."""
    return lib.HistoryItemDealToPlayer(self._item)

  def __str__(self):
    c_string = lib.HistoryItemToString(self._item)
    string = encode_ffi_string(c_string)
    lib.DeleteString(c_string)
    return string

  def __repr__(self):
    return self.__str__()

  def __del__(self):
    if self._item is not None:
      lib.DeleteHistoryItem(self._item)
      self._item = None
    del self


class HanabiEndOfGameType(enum.IntEnum):
  """Possible end-of-game conditions, consistent with hanabi_state.h."""
  NOT_FINISHED = 0
  OUT_OF_LIFE_TOKENS = 1
  OUT_OF_CARDS = 2
  COMPLETED_FIREWORKS = 3


class HanabiState(object):
  """Current environment state for an active Hanabi game.

  The game is turn-based, with only one active agent at a time. Chance events
  are explicitly included, so the active agent may be "nature" (represented
  by cur_player() returning CHANCE_PLAYER_ID).

  Python wrapper of C++ HanabiState class.
  """

  def __init__(self, game, c_state=None):
    """Returns a new state.

    Args:
      game: HanabiGame describing the parameters for a game of Hanabi.
      c_state: C++ state to copy, or None for a new state.

    NOTE: If c_state is supplied, game is ignored and c_state game is used.
    """
    self._state = ffi.new("pyhanabi_state_t*")
    if c_state is None:
      self._game = game.c_game
      lib.NewState(self._game, self._state)
    else:
      self._game = lib.StateParentGame(c_state)
      lib.CopyState(c_state, self._state)
    # MB: Create deck here too
    self._deck = HanabiDeck(game)

  def copy(self):
    """Returns a copy of the state."""
    return HanabiState(None, self._state)

  def observation(self, player):
    """Returns player's observed view of current environment state."""
    return HanabiObservation(self._state, self._game, player)

  def apply_move(self, move):
    """Advance the environment state by making move for acting player."""
    lib.StateApplyMove(self._state, move.c_move)

  def cur_player(self):
    """Returns index of next player to act.

    Index will be CHANCE_PLAYER_ID if a chance event needs to be resolved.
    """
    return lib.StateCurPlayer(self._state)

  def deck_size(self):
    """Returns number of cards left in the deck."""
    return lib.StateDeckSize(self._state)

  def discard_pile(self):
    """Returns a list of all discarded cards, in order they were discarded."""
    discards = []
    c_card = ffi.new("pyhanabi_card_t*")
    for index in range(lib.StateDiscardPileSize(self._state)):
      lib.StateGetDiscard(self._state, index, c_card)
      discards.append(HanabiCard(c_card.color, c_card.rank))
    return discards

  def fireworks(self):
    """Returns a list of fireworks levels by value, ordered by color (RYGWB).

    Important note on representation / format: when no fireworks have been
    played, this function returns [0, 0, 0, 0, 0]. When only the red 1 has been
    played, this function returns [1, 0, 0, 0, 0].
    """
    firework_list = []
    num_colors = lib.NumColors(self._game)
    for c in range(num_colors):
      firework_list.append(lib.StateFireworks(self._state, c))
    return firework_list

  def fireworks_score(self):
    """MB: Utility function. Return the combined fireworks score"""
    score=0
    fireworks = self.fireworks()
    for f in fireworks:
      score += f
    return score

  def deal_random_card(self):
    """If cur_player == CHANCE_PLAYER_ID, make a random card-deal move."""
    lib.StateDealCard(self._state)

  def deal_specific_card(self, color, rank, card_index):
    """MB: if cur_player = CHANCE_PLAYER_ID, make a specific card-deal move"""
    # Note: This move currently changes card knowledge. In the actual one, we don't want to change knowledge
    assert self.cur_player() == CHANCE_PLAYER_ID
    move = HanabiMove.get_deal_specific_move(color, rank, card_index)
    self.apply_move(move)

  def player_hands(self):
    """Returns a list of all hands, with cards ordered oldest to newest."""
    hand_list = []
    c_card = ffi.new("pyhanabi_card_t*")
    for pid in range(self.num_players()):
      player_hand = []
      hand_size = lib.StateGetHandSize(self._state, pid)
      for i in range(hand_size):
        lib.StateGetHandCard(self._state, pid, i, c_card)
        player_hand.append(HanabiCard(c_card.color, c_card.rank))
      hand_list.append(player_hand)
    return hand_list

  def information_tokens(self):
    """Returns the number of information tokens remaining."""
    return lib.StateInformationTokens(self._state)

  def end_of_game_status(self):
    """Returns the end of game status, NOT_FINISHED if game is still active."""
    return HanabiEndOfGameType(lib.StateEndOfGameStatus(self._state))

  def is_terminal(self):
    """Returns false if game is still active, true otherwise."""
    return (lib.StateEndOfGameStatus(self._state) !=
            HanabiEndOfGameType.NOT_FINISHED)

  def legal_moves(self):
    """Returns list of legal moves for currently acting player."""
    # MB: Work was needed to allow Return to be a valid move here.
    # MB: More work will be needed to make sure Agents don't think it's a valid move
    moves = []
    c_movelist = lib.StateLegalMoves(self._state)
    num_moves = lib.NumMoves(c_movelist)
    for i in range(num_moves):
      c_move = ffi.new("pyhanabi_move_t*")
      lib.GetMove(c_movelist, i, c_move)
      moves.append(HanabiMove(c_move))
    lib.DeleteMoveList(c_movelist)
    return moves

  def move_is_legal(self, move):
    """Returns true if and only if move is legal for active agent."""
    return lib.MoveIsLegal(self._state, move.c_move)

  def card_playable_on_fireworks(self, color, rank):
    """Returns true if and only if card can be successfully played.

    Args:
      color: 0-based color index of card
      rank: 0-based rank index of card
    """
    return lib.CardPlayableOnFireworks(self._state, color, rank)

  def life_tokens(self):
    """Returns the number of information tokens remaining."""
    return lib.StateLifeTokens(self._state)

  def num_players(self):
    """Returns the number of players in the game."""
    return lib.StateNumPlayers(self._state)

  def score(self):
    """Returns the co-operative game score at a terminal state.

    NOTE: result is undefined when game is NOT_FINISHED.
    """
    return lib.StateScore(self._state)

  def move_history(self):
    """Returns list of moves made, from oldest to most recent."""
    history = []
    history_len = lib.StateLenMoveHistory(self._state)
    for i in range(history_len):
      c_history_item = ffi.new("pyhanabi_history_item_t*")
      lib.StateGetMoveHistory(self._state, i, c_history_item)
      history.append(HanabiHistoryItem(c_history_item))
    return history

  def __str__(self):
    c_string = lib.StateToString(self._state)
    string = encode_ffi_string(c_string)
    lib.DeleteString(c_string)
    return string

  def __repr__(self):
    return self.__str__()

  def __del__(self):
    if self._state is not None:
      lib.DeleteState(self._state)
      self._state = None
    del self

  def valid_cards(self, player, card_index):
    """MB: Return list of HanabiCard that are a valid swap for the one questioned"""
    # Note: We know the state. For efficency and simplicity a direct GetDeck should have been implemented.
    # Then the only check needed is the card_knowledge check
    debug = False
    if debug: print("MB:  Replacing {}".format(self.player_hands()[player][card_index]))
    self._deck.reset_deck()

    # MB: First run through discard pile
    for card in self.discard_pile():
      self._deck.remove_card(card.color(), card.rank())
    if debug: print("MB: valid cards after discard: {}".format(self._deck))

    # MB: Then remove the cards that can be seen
    self._deck.remove_by_hands(player, card_index, self.player_hands())
    if debug: print("MB: valid cards after cards: {}".format(self._deck))

    # MB: Then remove cards that are making up fireworks
    self._deck.remove_by_fireworks(self.fireworks())
    if debug: print("MB: valid cards after fireworks: {}".format(self._deck))

    # MB: Finally use card knowledge player has about own hand from hints
    # MB: ! If retrieving something via C++ wrapper method need to assign to object first!
    temp_observation = self.observation(player)
    if debug: print("MB: valid_cards. Getting card_knowledge from player {} perspective".format(player))
    if debug: print("MB: valid cards attempting to access card_index {}".format(card_index))
    # MB: Is it assured that card_knowledge()[0] is the right call here?
    # Yes because the observation was player based.So the first will be the same as player
    card_knowledge = temp_observation.card_knowledge()[0][card_index]
    if debug: print("MB: valid cards retrieved card_knowledge")
    self._deck.remove_by_card_knowledge(card_knowledge)

    # MB: Return list of remaining cards in the deck; the valid options
    if debug: print("MB: Valid cards for player {} in position: {} are: {}".format(player,card_index,self._deck))
    return self._deck.return_cards()

  def valid_card(self, player, card_index):
    return random.choice(self.valid_cards(player, card_index))

  def replace_hand(self, player):
    """Redeterminise a player's hand based on valid permuation"""
    debug = False
    assert player == self.cur_player()
    for card_index in range(len(self.player_hands()[player])):
      # MB: There is a possibility when redeterminising that the hand ends up no longer valid
      valid_card = self.valid_card(player, card_index)
      if debug: print("MB: replace_hand will replace {} with {} for player {}".format(self.player_hands()[player][card_index],valid_card,player))
      assert valid_card is not None # MB: This is where it could slip up; intra-hand conflict
      return_move = HanabiMove.get_return_move(card_index=card_index)
      self.apply_move(return_move)
      if debug: print("MB: replace_hand passed return move")
      # Check where this is dealt to
      deal_specific_move = HanabiMove.get_deal_specific_move(card_index, player, valid_card.color(), valid_card.rank())
      self.apply_move(deal_specific_move)
      if debug: print("MB: replace_hand successfully replaced that card")


class HanabiDeck(object):
  """MB: Seperate class handling Python level deck functions for forward models"""
  # Store deck for easier theoretical manipulation

  def __init__(self, _game):
    self.debug = False
    self.num_ranks_ = _game.num_ranks()
    self.num_colors_ = _game.num_colors()
    self.num_cards = _game.num_cards
    self.card_count_ = []  # Card count entries are number 0 - 3, how many of card_index index there are in deck
    self.total_count_ = 0  # total cards in deck
    self.reset_deck()

  def reset_deck(self):
    self.card_count_ = []
    self.total_count_ = 0
    # MB:Iteration is in same format as card_to_index, so fine to append (ORRR ARE WEEEE?)
    for color in range(self.num_colors_):
      for rank in range(self.num_ranks_):
        # MB: Num cards accounts for duplicate numbers for each card
        count = self.num_cards(color, rank)
        self.card_count_.append(count)
        self.total_count_ += count

  def remove_by_card_knowledge(self, card_knowledge):
    """MB: Remove all cards from deck that don't fit with the card knowledge"""
    for color in range(self.num_colors_):
      for rank in range(self.num_ranks_):
        if not (card_knowledge.color_plausible(color) and card_knowledge.rank_plausible(rank)):
          self.remove_all_card(color, rank)

  def remove_by_hands(self, player, card_index, hands):
    for other_player in range(len(hands)):
      for card_index_i in range(len(hands[other_player])):
        # MB: For now, also peek at rest of own hand and remove all those not at card_index
        if other_player == player and card_index_i == card_index:
          continue  # skip over the player's card that we ask for valid cards for.
        card = hands[other_player][card_index_i]
        self.remove_card(card.color(), card.rank())

  def remove_by_fireworks(self, fireworks):
    """MB: Remove all cards from deck that are making up fireworks"""
    for color in range(self.num_colors_):
        for rank in range(fireworks[color]):
          self.remove_card(color, rank)

  def remove_card(self, color, rank):
    card_index_ = self.card_to_index(color, rank)
    if self.card_count_[card_index_] <= 0:
      print("MB: Warning! pyhanabi.HanabiDeck.remove_card: Card color: {}, rank {} not removed as not in deck."
            " HanabiState.valid_cards likely failed.".format(color, rank))
      return
    self.card_count_[card_index_] -= 1
    self.total_count_ -= 1

  def remove_all_card(self, color, rank):
    """MB: Removes all instances of particular card index. Silently in current form"""
    card_index = self.card_to_index(color, rank)
    while self.card_count_[card_index] > 0:
      self.remove_card(color, rank)

  def return_cards(self):
    """MB: Return the deck as HanabiCard objects"""
    cards = []
    for color in range(self.num_colors_):
      for rank in range(self.num_ranks_):
        card_index = self.card_to_index(color, rank)
        for _ in range(self.card_count_[card_index]):
          cards.append(HanabiCard(color, rank))
    return cards

  def card_to_index(self, color, rank):
    return color * self.num_ranks_ + rank

  def is_empty(self):
    return self.total_count_ == 0

  def size(self):
    return

  def __str__(self):
    deck_string = ""
    cards = self.return_cards()
    for card in cards:
      deck_string += " {}".format(card)
    return deck_string


class AgentObservationType(enum.IntEnum):
  """Possible agent observation types, consistent with hanabi_game.h.

  A kMinimal observation is similar to what a human sees, and does not
  include any memory of past RevalColor/RevealRank hints. A CardKnowledge
  observation includes per-card knowledge of past hints, as well as simple
  inferred knowledge of the form "this card is not red, because it was
  not revealed as red in a past <RevealColor Red> move. A Seer observation
  shows all cards, including the player's own cards, regardless of what
  hints have been given.
  """
  MINIMAL = 0
  CARD_KNOWLEDGE = 1
  SEER = 2

class HanabiGame(object):
  """Game parameters describing a specific instance of Hanabi.

  Python wrapper of C++ HanabiGame class.
  """

  def __init__(self, params=None):
    """Creates a HanabiGame object.

    Args:
      params: is a dictionary of parameters and their values.

    Possible parameters include
    "players": 2 <= number of players <= 5
    "colors": 1 <= number of different card colors in deck <= 5
    "rank": 1 <= number of different card ranks in deck <= 5
    "hand_size": 1 <= number of cards in player hand
    "max_information_tokens": 1 <= maximum (and initial) number of info tokens.
    "max_life_tokens": 1 <= maximum (and initial) number of life tokens.
    "seed": random number seed. -1 to use system random device to get seed.
    "random_start_player": boolean. If true, start with random player, not 0.
    "observation_type": int AgentObservationType.
    """
    if params is None:
      self._game = ffi.new("pyhanabi_game_t*")
      lib.NewDefaultGame(self._game)
    else:
      param_list = []
      for key in params:
        param_list.append(ffi.new("char[]", key.encode('ascii')))
        param_list.append(ffi.new("char[]", str(params[key]).encode('ascii')))
      c_array = ffi.new("char * [" + str(len(param_list)) + "]", param_list)
      self._game = ffi.new("pyhanabi_game_t*")
      lib.NewGame(self._game, len(param_list), c_array)

  def new_initial_state(self):
    return HanabiState(self)

  @property
  def c_game(self):
    """Return the C++ HanabiGame object."""
    return self._game

  def __del__(self):
    if self._game is not None:
      lib.DeleteGame(self._game)
      self._game = None
    del self

  def parameter_string(self):
    """Returns string with all parameter choices."""
    c_string = lib.GameParamString(self._game)
    string = encode_ffi_string(c_string)
    lib.DeleteString(c_string)
    return string

  def num_players(self):
    """Returns the number of players in the game."""
    return lib.NumPlayers(self._game)

  def num_colors(self):
    """Returns the number of card colors in the initial deck."""
    return lib.NumColors(self._game)

  def num_ranks(self):
    """Returns the number of card ranks in the initial deck."""
    return lib.NumRanks(self._game)

  def hand_size(self):
    """Returns the maximum number of cards in each player hand.

    The number of cards in a player's hand may be smaller than this maximum
    a) at the beginning of the game before cards are dealt out, b) after
    any Play or Discard action and before the subsequent deal event, and c)
    after the deck is empty and cards can no longer be dealt to a player.
    """
    return lib.HandSize(self._game)

  def max_information_tokens(self):
    """Returns the initial number of information tokens."""
    return lib.MaxInformationTokens(self._game)

  def max_life_tokens(self):
    """Returns the initial number of life tokens."""
    return lib.MaxLifeTokens(self._game)

  def observation_type(self):
    return AgentObservationType(lib.ObservationType(self._game))

  def max_moves(self):
    """Returns the number of possible legal moves in the game."""
    return lib.MaxMoves(self._game)

  def num_cards(self, color, rank):
    """Returns number of instances of Card(color, rank) in the initial deck."""
    return lib.NumCards(self._game, color, rank)

  def get_move_uid(self, move):
    """Returns a unique ID describing a legal move, or -1 for invalid move."""
    return lib.GetMoveUid(self._game, move.c_move)

  def get_move(self, move_uid):
    """Returns a HanabiMove represented by 0 <= move_uid < max_moves()."""
    move = ffi.new("pyhanabi_move_t*")
    lib.GetMoveByUid(self._game, move_uid, move)
    return HanabiMove(move)


class HanabiObservation(object):
  """Player's observed view of an environment HanabiState.

  The main differences are that 1) a player's own cards are not visible, and
  2) a player does not know their own player index (seat) so that all player
  indices are described relative to the observing player (or equivalently,
  that from the player's point of view, they are always player index 0).

  Python wrapper of C++ HanabiObservation class.
  """

  def __init__(self, state, game, player):
    """Construct using HanabiState.observation(player)."""
    self._observation = ffi.new("pyhanabi_observation_t*")
    self._game = game
    lib.NewObservation(state, player, self._observation)

  def __str__(self):
    c_string = lib.ObsToString(self._observation)
    string = encode_ffi_string(c_string)
    lib.DeleteString(c_string)
    return string

  def __repr__(self):
    return self.__str__()

  def __del__(self):
    if self._observation is not None:
      lib.DeleteObservation(self._observation)
      self._observation = None
    del self

  def observation(self):
    """Returns the C++ HanabiObservation object."""
    return self._observation

  def cur_player_offset(self):
    """Returns the player index of the acting player, relative to observer."""
    return lib.ObsCurPlayerOffset(self._observation)

  def num_players(self):
    """Returns the number of players in the game."""
    return lib.ObsNumPlayers(self._observation)

  def observed_hands(self):
    """Returns a list of all hands, with cards ordered oldest to newest.

     The observing player's cards are always invalid.
    """
    hand_list = []
    c_card = ffi.new("pyhanabi_card_t*")
    for pid in range(self.num_players()):
      player_hand = []
      hand_size = lib.ObsGetHandSize(self._observation, pid)
      for i in range(hand_size):
        lib.ObsGetHandCard(self._observation, pid, i, c_card)
        player_hand.append(HanabiCard(c_card.color, c_card.rank))
      hand_list.append(player_hand)
    return hand_list

  def card_knowledge(self):
    """Returns a per-player list of hinted card knowledge.

    Each player's entry is a per-card list of HanabiCardKnowledge objects.
    Each HanabiCardKnowledge for a card gives the knowledge about the cards
    accumulated over all past reveal actions.
    """
    debug = False
    card_knowledge_list = []
    for pid in range(self.num_players()):
      player_card_knowledge = []
      # MB: This gets desynched after retrun/deal specific move
      hand_size = lib.ObsGetHandSize(self._observation, pid)
      for i in range(hand_size):
        c_knowledge = ffi.new("pyhanabi_card_knowledge_t*")
        #if debug: print("MB: card_knowledge() trying to retrieve CardKnowledge for {}".format(pid))
        c_card = ffi.new("pyhanabi_card_t*")
        lib.ObsGetHandCard(self._observation, pid, i, c_card)
        lib.ObsGetHandCardKnowledge(self._observation, pid, i, c_knowledge)
        player_card_knowledge.append(HanabiCardKnowledge(c_knowledge))
        if debug: print("MB: card_knowledge {} for card in hand {} retrieved for pid {}, card {}, hand_size {}".format(
          HanabiCardKnowledge(c_knowledge),HanabiCard(c_card.color,c_card.rank), pid ,i ,hand_size))
      card_knowledge_list.append(player_card_knowledge)
    return card_knowledge_list

  def discard_pile(self):
    """Returns a list of all discarded cards, in order they were discarded."""
    discards = []
    c_card = ffi.new("pyhanabi_card_t*")
    for index in range(lib.ObsDiscardPileSize(self._observation)):
      lib.ObsGetDiscard(self._observation, index, c_card)
      discards.append(HanabiCard(c_card.color, c_card.rank))
    return discards

  def fireworks(self):
    """Returns a list of fireworks levels by value, ordered by color."""
    firework_list = []
    num_colors = lib.NumColors(self._game)
    for c in range(num_colors):
      firework_list.append(lib.ObsFireworks(self._observation, c))
    return firework_list

  def deck_size(self):
    """Returns number of cards left in the deck."""
    return lib.ObsDeckSize(self._observation)

  def last_moves(self):
    """Returns moves made since observing player last acted.

    Each entry in list is a HanabiHistoryItem, ordered from most recent
    move to oldest.  Oldest move is the last action made by observing
    player. Skips initial chance moves to deal hands.
    """
    history_items = []
    for i in range(lib.ObsNumLastMoves(self._observation)):
      history_item = ffi.new("pyhanabi_history_item_t*")
      lib.ObsGetLastMove(self._observation, i, history_item)
      history_items.append(HanabiHistoryItem(history_item))
    return history_items

  def information_tokens(self):
    """Returns the number of information tokens remaining."""
    return lib.ObsInformationTokens(self._observation)

  def life_tokens(self):
    """Returns the number of information tokens remaining."""
    return lib.ObsLifeTokens(self._observation)

  def legal_moves(self):
    """Returns list of legal moves for observing player.

    List is empty if cur_player() != 0 (observer is not currently acting).
    """
    moves = []
    for i in range(lib.ObsNumLegalMoves(self._observation)):
      move = ffi.new("pyhanabi_move_t*")
      lib.ObsGetLegalMove(self._observation, i, move)
      moves.append(HanabiMove(move))
    return moves

  def card_playable_on_fireworks(self, color, rank):
    """Returns true if and only if card can be successfully played.

    Args:
      color: 0-based color index of card
      rank: 0-based rank index of card
    """
    return lib.ObsCardPlayableOnFireworks(self._observation, color, rank)


class ObservationEncoderType(enum.IntEnum):
  """Encoder types, consistent with observation_encoder.h."""
  CANONICAL = 0


class ObservationEncoder(object):
  """ObservationEncoder class.

  The canonical observations wrap an underlying C++ class. To make custom
  observation encoders, create a subclass of this base class and override
  the shape and encode methods.
  """

  def __init__(self, game, enc_type=ObservationEncoderType.CANONICAL):
    """Construct using HanabiState.observation(player)."""
    self._game = game.c_game
    self._encoder = ffi.new("pyhanabi_observation_encoder_t*")
    lib.NewObservationEncoder(self._encoder, self._game, enc_type)

  def __del__(self):
    if self._encoder is not None:
      lib.DeleteObservationEncoder(self._encoder)
      self._encoder = None
      self._game = None
    del self

  def shape(self):
    c_shape_str = lib.ObservationShape(self._encoder)
    shape_string = encode_ffi_string(c_shape_str)
    lib.DeleteString(c_shape_str)
    shape = [int(x) for x in shape_string.split(",")]
    return shape

  def encode(self, observation):
    """Encode the observation as a sequence of bits."""
    c_encoding_str = lib.EncodeObservation(self._encoder,
                                           observation.observation())
    print("MB: EncodeObservation success")
    encoding_string = encode_ffi_string(c_encoding_str)
    lib.DeleteString(c_encoding_str)
    # Canonical observations are bit strings, so it is ok to encode using a
    # string. For float or double observations, make a custom object
    encoding = [int(x) for x in encoding_string.split(",")]
    return encoding


try_cdef()
if cdef_loaded():
  try_load()
