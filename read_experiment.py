# Designed for storing and reading experiment data
import pandas as pd

class Experiment():
  def __init__(self,flags,stats_keys,scores,game_stats,player_stats,avg_score,avg_time,game_stats_full=None,player_stats_full=None):
    self.flags = flags
    self.scores = scores
    self.stats_keys = stats_keys
    self.game_stats = game_stats #lsit
    self.game_stats_full = game_stats_full
    self.player_stats = player_stats
    self.player_stats_full = player_stats_full
    self.avg_score = avg_score
    self.avg_time = avg_time

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
  for k,v in file_names.items():
    experiments = None
    print(f"Readin {k}")
    exec(open(v).read()) # Defines a list of experiments (eg. 2p - 5p)
    assert experiments is not None
    master_experiments[k] = experiments

  explain_experiments(master_experiments["vrun_vdb"])


