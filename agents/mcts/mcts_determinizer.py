# MCTSDetermizer: Handles determinising and redeterminising player hands
from pyhanabi import HanabiCard
import random

class MCTSDeterminizer(object):
  def __init__(self):
    self.deck = HanabiDeck()

  def valid_card(self, player, card_index, player_hands, discard_pile, fireworks, card_knowledge):
    return random.choice(self.valid_cards(player, card_index, player_hands, discard_pile, fireworks, card_knowledge))

  def valid_cards(self, player, card_index, player_hands, discard_pile, fireworks, card_knowledge):
    """MB: Return list of HanabiCard that are a valid swap for the one questioned"""
    # Note: We know the state. For efficency and simplicity a direct GetDeck should have been implemented.
    # Then the only check needed is the card_knowledge check
    debug = False
    # if debug: print("MB:  Replacing {}".format(self.player_hands()[player][card_index]))
    self.deck.reset_deck()

    # MB: First run through discard pile
    for card in discard_pile:
      self.deck.remove_card(card.color(), card.rank())
    if debug: print("MB: valid cards after discard: {}".format(self.deck))

    # MB: Then remove the cards that can be seen
    self.deck.remove_by_hands(player, card_index, player_hands)
    if debug: print("MB: valid cards after cards: {}".format(self.deck))

    # MB: Then remove cards that are making up fireworks
    self.deck.remove_by_fireworks(fireworks)
    if debug: print("MB: valid cards after fireworks: {}".format(self.deck))

    # MB: Finally use card knowledge player has about own hand from hints
    self.deck.remove_by_card_knowledge(card_knowledge[card_index])

    # MB: Return list of remaining cards in the deck; the valid options
    if debug: print("MB: Valid cards for player {} in position: {} are: {}".format(player, card_index, self.deck))
    return self.deck.get_deck()


class HanabiDeck(object):
  """MB: Seperate class handling Python level deck functions for forward models"""
  # Store deck for easier theoretical manipulation

  def __init__(self):
    self.debug = False
    # Hack to get around _game HanabiState issues
    self.num_ranks_ = 5
    self.num_colors_ = 5
    self.card_count_ = []  # Card count entries are number 0 - 3, how many of card_index index there are in deck
    self.total_count_ = 0  # total cards in deck
    self.reset_deck()

  def num_cards(self, color, rank):
    # MB: Hack to get around _game HanabiState issues
    assert rank >= 0 and rank <= 4
    if rank == 0:
      return 3
    elif rank == 4:
      return 1
    else:
      return 2

  def reset_deck(self):
    self.card_count_ = []
    self.total_count_ = 0
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

  def get_deck(self):
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
    cards = self.get_deck()
    for card in cards:
      deck_string += " {}".format(card)
    return deck_string
