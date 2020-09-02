from rl_env import Agent
from agents.rule_based.rule_based_agents import VanDenBerghAgent
from agents.rule_based.rule_based_agents import OuterAgent
from agents.rule_based.rule_based_agents import InnerAgent
from agents.rule_based.rule_based_agents import PiersAgent
from agents.rule_based.rule_based_agents import IGGIAgent
from agents.rule_based.rule_based_agents import LegalRandomAgent
from agents.rule_based.rule_based_agents import FlawedAgent
from agents.rule_based.rule_based_agents import MuteAgent

AGENT_CLASSES = {'VANDENBERGH': VanDenBerghAgent,'FLAWEDAGENT':FlawedAgent
                  , 'OUTERAGENT':OuterAgent, 'INNERAGENT':InnerAgent, 'PIERSAGENT':PiersAgent, 'IGGIAGENT':IGGIAgent
                  , 'LEGALRANDOMAGENT':LegalRandomAgent, 'MUTEAGENT':MuteAgent}


class HumanAgent(Agent):
  def __init__(self,config):
    self.max_information_tokens = config.get('information_tokens', 8)
    self.agent = None
    self.config = config

  def print_all_moves(self, legal_actions):
    for action in legal_actions:
      for param,value in action.items():
        print(value, end=" ")
      print(",", end="")


  def act(self, observation):
    # If not my turn, return nothing
    if observation['current_player_offset'] != 0:
      return None

    for move in reversed(observation["pyhanabi"].last_moves()):
      print(move)
    print(observation["pyhanabi"])

    if self.agent is not None:
      return self.agent.act(observation)

    legal_actions = observation["legal_moves"]

    while True:
      action = {}
      # Get user input
      print("Possible actions: ", end="")
      self.print_all_moves(legal_actions)
      print("")
      human_input = input("Choose an action, or bot followed by an agent:")
      action_args = [a.upper() for a in human_input.split(" ")]
      action['action_type'] = action_args[0]

      if action['action_type'] == 'BOT':
        if len(action_args) <= 1:
          print("Bot needs a valid agent name")
        if action_args[1] in AGENT_CLASSES:
          self.agent = AGENT_CLASSES[action_args[1]](self.config)
          return self.agent.act(observation)
        else:
          print(f"Not a valid agent. Choose {[a for a in AGENT_CLASSES.keys()]}")

      if action['action_type'] == 'PLAY' or action['action_type'] == 'DISCARD':
        if len(action_args) <= 1:
          print("Seperate all action arguments with spaces")
        elif action_args[1].isdigit():
          action['card_index'] = int(action_args[1])
      elif action['action_type'] == 'REVEAL_RANK' or action['action_type'] == 'REVEAL_COLOR':
        if len(action_args) <= 2:
          print("Seperate all action arguments with spaces")
        elif action['action_type'] == 'REVEAL_RANK':
          if action_args[1].isdigit():
            action['target_offset'] = int(action_args[1])
          if action_args[2].isdigit():
            action['rank'] = int(action_args[2])-1
        elif action['action_type'] == 'REVEAL_COLOR':
          if action_args[1].isdigit():
            action['target_offset'] = int(action_args[1])
          action['color'] = action_args[2]

      if action in legal_actions:
        return action
      else:
        print(f"Action is not legal here: {action}")



