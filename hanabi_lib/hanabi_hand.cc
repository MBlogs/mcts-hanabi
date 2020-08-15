// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hanabi_hand.h"
#include <iostream>
#include <algorithm>
#include <cassert>

#include "util.h"

namespace hanabi_learning_env {

HanabiHand::ValueKnowledge::ValueKnowledge(int value_range)
    : value_(-1), value_plausible_(std::max(value_range, 0), true) {
  assert(value_range > 0);
}

void HanabiHand::ValueKnowledge::ApplyIsValueHint(int value) {
  assert(value >= 0 && value < value_plausible_.size());
  assert(value_ < 0 || value_ == value);
  assert(value_plausible_[value] == true);
  value_ = value;
  std::fill(value_plausible_.begin(), value_plausible_.end(), false);
  value_plausible_[value] = true;
}

void HanabiHand::ValueKnowledge::ApplyIsNotValueHint(int value) {
  assert(value >= 0 && value < value_plausible_.size());
  assert(value_ < 0 || value_ != value);
  value_plausible_[value] = false;
}

HanabiHand::CardKnowledge::CardKnowledge(int num_colors, int num_ranks)
    : color_(num_colors), rank_(num_ranks) {}

std::string HanabiHand::CardKnowledge::ToString() const {
  std::string result;
  result = result + (ColorHinted() ? ColorIndexToChar(Color()) : 'X') +
           (RankHinted() ? RankIndexToChar(Rank()) : 'X') + '|';
  //MB: if it goes over there is a memory leak
  assert(color_.Range() <= 8);

  for (int c = 0; c < color_.Range(); ++c) {

    if (color_.IsPlausible(c)) {
      result += ColorIndexToChar(c);
    }
  }

  for (int r = 0; r < rank_.Range(); ++r) {
    if (rank_.IsPlausible(r)) {
      result += RankIndexToChar(r);
    }
  }
  return result;
}

HanabiHand::HanabiHand(const HanabiHand& hand, bool hide_cards,
                       bool hide_knowledge) {
  if (hide_cards) {
    cards_.resize(hand.cards_.size(), HanabiCard());
  } else {
    cards_ = hand.cards_;
  }
  if (hide_knowledge && !hand.cards_.empty()) {
    card_knowledge_.resize(hand.cards_.size(),
                           CardKnowledge(hand.card_knowledge_[0].NumColors(),
                                         hand.card_knowledge_[0].NumRanks()));
  } else {
    card_knowledge_ = hand.card_knowledge_;
  }
}

void HanabiHand::AddCard(HanabiCard card,
                         const CardKnowledge& initial_knowledge) {
  REQUIRE(card.IsValid());
  cards_.push_back(card);
  card_knowledge_.push_back(initial_knowledge);
}

void HanabiHand::RemoveFromHand(int card_index,
                                std::vector<HanabiCard>* discard_pile) {
  if (discard_pile != nullptr) {
    discard_pile->push_back(cards_[card_index]);
  }
  cards_.erase(cards_.begin() + card_index);
  card_knowledge_.erase(card_knowledge_.begin() + card_index);
}

void HanabiHand::InsertCard(HanabiCard card,int card_index) {
  // MB: No reset of card knowledge and insertion of card into hand
  //MB: The choise of Insert
  REQUIRE(card.IsValid());
  cards_.insert(cards_.begin()+card_index,card);
}

void HanabiHand::ReturnFromHand(int card_index) {
  // MB: Delete from hand (and try to not delete card knowledge too)
  // MB: Adding to deck is handled by ApplyMove in hanabi_state
  // MB: cards_ is vector<HanabiCard> , card_knowledge_ is vector<HanabiKnowledge>
  cards_.erase(cards_.begin() + card_index);
}

void HanabiHand::RemoveKnowledge(int card_index, const CardKnowledge& initial_knowledge) {
    // Stand alone: Remove card knowledge only
    // (use case: eg. used ReturnCard because we thought we wanted to retain knowledge.
    // but we now realise we want to remove it instead
    card_knowledge_.erase(card_knowledge_.begin() + card_index);
    // Similar to cards, we insert the default knowledge in
    card_knowledge_.insert(card_knowledge_.begin()+card_index,initial_knowledge);
}

uint8_t HanabiHand::RevealColor(const int color) {
  uint8_t mask = 0;
  assert(cards_.size() <= 8);  // More than 8 cards is currently not supported.
  for (int i = 0; i < cards_.size(); ++i) {
    if (cards_[i].Color() == color) {
      if (!card_knowledge_[i].ColorHinted()) {
        mask |= static_cast<uint8_t>(1) << i;
      }
      card_knowledge_[i].ApplyIsColorHint(color);
    } else {
      card_knowledge_[i].ApplyIsNotColorHint(color);
    }
  }
  return mask;
}

uint8_t HanabiHand::RevealRank(const int rank) {
  uint8_t mask = 0;
  assert(cards_.size() <= 8);  // More than 8 cards is currently not supported.
  for (int i = 0; i < cards_.size(); ++i) {
    if (cards_[i].Rank() == rank) {
      if (!card_knowledge_[i].RankHinted()) {
        mask |= static_cast<uint8_t>(1) << i;
      }
      card_knowledge_[i].ApplyIsRankHint(rank);
    } else {
      card_knowledge_[i].ApplyIsNotRankHint(rank);
    }
  }
  return mask;
}

std::string HanabiHand::ToString() const {
  std::string result;
  assert(cards_.size() == card_knowledge_.size());
  for (int i = 0; i < cards_.size(); ++i) {
    result +=
        cards_[i].ToString() + " || " + card_knowledge_[i].ToString() + '\n';
  }
  return result;
}

}  // namespace hanabi_learning_env
