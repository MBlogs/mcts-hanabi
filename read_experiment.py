# Designed for storing and reading experiment data
import pandas as pd

class Experiment():
  def __init__(self,flags,mcts_configs,stats_keys,scores,game_stats,player_stats,avg_score,avg_time,errors,game_stats_full=None,player_stats_full=None):
    self.flags = flags
    self.mcts_configs = mcts_configs
    self.scores = scores
    self.stats_keys = stats_keys
    self.game_stats = game_stats #lsit
    self.game_stats_full = game_stats_full
    self.player_stats = player_stats
    self.player_stats_full = player_stats_full
    self.avg_score = avg_score
    self.avg_time = avg_time
    self.errors = errors

  def compute_avg_stat(self,stat):
    total_score = sum([game[self.stat(stat)] for game in self.game_stats])
    return total_score/len(self.game_stats)

  def convert_to_df(self):
    game_df = pd.DataFrame({stat:[s[self.stat(stat)] for s in self.game_stats] for stat in self.stats_keys})
    return game_df

  def stat(self,key_string):
    return self.stats_keys.index(key_string)

  def __str__(self):
    return str(self.flags)

  def __repr__(self):
    return str(self.flags)


def explain_experiments(exps):
  for exp in exps:
    print(exp)
    print(f"Available stats: {exp.stats_keys}")
    print(f"Game stats: {exp.game_stats}")
    print(f"Printed average game score {exp.avg_score}")
    print(f"Computed average game score {exp.compute_avg_stat('score')}")
    print(f"Computed average game regret {exp.compute_avg_stat('regret')}")
    print(f"Computed average discard critical {exp.compute_avg_stat('discard_critical')}")
    print(f"Computed average play fail {exp.compute_avg_stat('discard_critical')}")
    # print(exp.convert_to_df())

if __name__ == "__main__":
  file_names = {"vrun_vdb":"vrun_vdb.sh.o1214497"
    , "vrun_random":"vrun_random.sh.o1214553"
    , "vrun_iggi":"vrun_iggi.sh.o1214560"
    , "vrun_flawed": "vrun_flawed.sh.o1214580"
  }

  master_experiments = {}
  #for k,v in file_names.items():
  #  experiments = None
  #   print(f"Readin {k}")
  #  # exec(open(v).read()) # Defines a list of experiments (eg. 2p - 5p)
  #  assert experiments is not None
  #  master_experiments[k] = experiments
  experiments = [Experiment(
    flags={'players': 3, 'num_episodes': 2, 'agent': 'MCTSAgent', 'agents': 'MCTSAgent', 'mcts_types': 'xxx',
           'agent_classes': ['MCTSAgent', 'MCTSAgent', 'MCTSAgent']}
    , mcts_configs=[
      {'max_time_limit': 1000, 'max_rollout_num': 100,
       'agents': '[<agents.rule_based.rule_based_agents.VanDenBerghAgent object at 0x7f8d79fdd760>, <agents.rule_based.rule_based_agents.VanDenBerghAgent object at 0x7f8d79fdd850>, <agents.rule_based.rule_based_agents.VanDenBerghAgent object at 0x7f8d7232dee0>]',
       'max_simulation_steps': 3, 'max_depth': 100, 'determine_type': 0, 'score_type': 0, 'exploration_weight': 2.5,
       'rules': '[<function Ruleset.tell_most_information_factory.<locals>.tell_most_information at 0x7f8d72330280>, <function Ruleset.tell_anyone_useful_card at 0x7f8d7231ea60>, <function Ruleset.tell_dispensable_factory.<locals>.tell_dispensable at 0x7f8d723309d0>, <function Ruleset.complete_tell_useful at 0x7f8d723240d0>, <function Ruleset.complete_tell_dispensable at 0x7f8d72324160>, <function Ruleset.complete_tell_unplayable at 0x7f8d723241f0>, <function Ruleset.play_probably_safe_factory.<locals>.play_probably_safe_treshold at 0x7f8d72330a60>, <function Ruleset.play_probably_safe_late_factory.<locals>.play_probably_safe_late at 0x7f8d72330af0>, <function Ruleset.discard_most_confident at 0x7f8d72324310>]'},
      {'max_time_limit': 1000, 'max_rollout_num': 100,
       'agents': '[<agents.rule_based.rule_based_agents.VanDenBerghAgent object at 0x7f8d79f71550>, <agents.rule_based.rule_based_agents.VanDenBerghAgent object at 0x7f8d79f71610>, <agents.rule_based.rule_based_agents.VanDenBerghAgent object at 0x7f8d79f718e0>]',
       'max_simulation_steps': 3, 'max_depth': 100, 'determine_type': 0, 'score_type': 0, 'exploration_weight': 2.5,
       'rules': '[<function Ruleset.tell_most_information_factory.<locals>.tell_most_information at 0x7f8d72330b80>, <function Ruleset.tell_anyone_useful_card at 0x7f8d7231ea60>, <function Ruleset.tell_dispensable_factory.<locals>.tell_dispensable at 0x7f8d79f73310>, <function Ruleset.complete_tell_useful at 0x7f8d723240d0>, <function Ruleset.complete_tell_dispensable at 0x7f8d72324160>, <function Ruleset.complete_tell_unplayable at 0x7f8d723241f0>, <function Ruleset.play_probably_safe_factory.<locals>.play_probably_safe_treshold at 0x7f8d79f733a0>, <function Ruleset.play_probably_safe_late_factory.<locals>.play_probably_safe_late at 0x7f8d79f73430>, <function Ruleset.discard_most_confident at 0x7f8d72324310>]'},
      {'max_time_limit': 1000, 'max_rollout_num': 100,
       'agents': '[<agents.rule_based.rule_based_agents.VanDenBerghAgent object at 0x7f8d79f71e20>, <agents.rule_based.rule_based_agents.VanDenBerghAgent object at 0x7f8d79f71fd0>, <agents.rule_based.rule_based_agents.VanDenBerghAgent object at 0x7f8d79f75220>]',
       'max_simulation_steps': 3, 'max_depth': 100, 'determine_type': 0, 'score_type': 0, 'exploration_weight': 2.5,
       'rules': '[<function Ruleset.tell_most_information_factory.<locals>.tell_most_information at 0x7f8d79f734c0>, <function Ruleset.tell_anyone_useful_card at 0x7f8d7231ea60>, <function Ruleset.tell_dispensable_factory.<locals>.tell_dispensable at 0x7f8d79f73c10>, <function Ruleset.complete_tell_useful at 0x7f8d723240d0>, <function Ruleset.complete_tell_dispensable at 0x7f8d72324160>, <function Ruleset.complete_tell_unplayable at 0x7f8d723241f0>, <function Ruleset.play_probably_safe_factory.<locals>.play_probably_safe_treshold at 0x7f8d79f73ca0>, <function Ruleset.play_probably_safe_late_factory.<locals>.play_probably_safe_late at 0x7f8d79f73d30>, <function Ruleset.discard_most_confident at 0x7f8d72324310>]'},
    ]
    , scores=[21, 15, ]
    ,
    stats_keys=['score', 'moves', 'regret', 'regret_discard_critical', 'regret_play_fail', 'regret_play_fail_critical',
                'regret_play_fail_endgame', 'discard', 'discard_critical', 'discard_useful', 'discard_safe', 'play',
                'play_success', 'play_fail', 'play_fail_critical', 'play_fail_endgame', 'information',
                'information_color', 'information_rank', 'elapsed_time']
    , game_stats=[[21, 63, 3, 3, 0, 0, 0, 16, 1, 4, 11, 21, 21, 0, 0, 0, 26, 10, 16, 62955],
                  [15, 64, 5, 5, 0, 0, 0, 19, 2, 8, 9, 17, 15, 2, 0, 0, 28, 12, 16, 64203]]
    , player_stats=[[[21, 21, 0, 0, 0, 0, 0, 6, 0, 0, 6, 5, 5, 0, 0, 0, 10, 5, 5, 21204],
                     [15, 22, 0, 0, 0, 0, 0, 9, 0, 4, 5, 4, 4, 0, 0, 0, 9, 3, 6, 21871]],
                    [[21, 21, 3, 3, 0, 0, 0, 7, 1, 2, 4, 8, 8, 0, 0, 0, 6, 1, 5, 20972],
                     [14, 21, 5, 5, 0, 0, 0, 7, 2, 4, 1, 7, 6, 1, 0, 0, 7, 4, 3, 21354]],
                    [[21, 21, 0, 0, 0, 0, 0, 3, 0, 2, 1, 8, 8, 0, 0, 0, 10, 4, 6, 20779],
                     [14, 21, 0, 0, 0, 0, 0, 3, 0, 0, 3, 6, 5, 1, 0, 0, 12, 5, 7, 20978]]]
    , avg_score=18.0
    , avg_time=1001.9253246753246
    , errors=0
  ),
  ]
  assert experiments is not None
  master_experiments["temp"] = experiments
  explain_experiments(master_experiments["temp"])


