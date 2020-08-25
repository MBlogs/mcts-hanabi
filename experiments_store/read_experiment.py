# Designed for storing and reading experiment data
import pandas as pd
import matplotlib.pyplot as plt

class Experiment():
  def __init__(self,flags,mcts_configs,stats_keys,scores,progress,game_stats,player_stats,avg_score,avg_progress
               ,avg_time,errors,game_stats_full=None,player_stats_full=None):
    self.flags = flags
    self.mcts_configs = mcts_configs
    self.scores = scores
    self.progress = progress
    self.stats_keys = stats_keys
    self.game_stats = game_stats #lsit
    self.game_stats_full = game_stats_full
    self.player_stats = player_stats
    self.player_stats_full = player_stats_full
    self.avg_score = avg_score
    self.avg_time = avg_time
    self.errors = errors
    self.game_df = self.convert_to_df(game_stats)
    self.player_dfs = []
    for i in range(len(self.player_stats)):
      self.player_dfs.append(self.convert_to_df(player_stats[i]))

  def compute_avg_stat(self,stat):
    total_score = sum([game[self.stat(stat)] for game in self.game_stats])
    return total_score/len(self.game_stats)

  def convert_to_df(self, stat_list):
    stat_df = pd.DataFrame({stat_string:[s[self.stat(stat_string)] for s in stat_list] for stat_string in self.stats_keys})
    return stat_df

  def stat(self,stat_string):
    return self.stats_keys.index(stat_string)

  def __str__(self):
    return str(self.flags)

  def __repr__(self):
    return str(self.flags)

def plot_score_hist(exp, series,xlabel="x",ylabel="y",title="",xlim=(0,25),ylim=None):
  plt.hist(series)
  plt.xlabel = xlabel
  plt.ylabel = ylabel
  plt.xlim = xlim
  if ylim:
    plt.ylim = ylim
  plt.title = title
  plt.show()

def explain_experiment(exp):
  print(exp)
  print(f"Available stats: {exp.stats_keys}")
  print(f"Game stats: {exp.game_stats}")
  print(f"Printed average game score {exp.avg_score}")
  print(f"Computed average game score {exp.compute_avg_stat('score')}")
  print(f"Computed average game regret {exp.compute_avg_stat('regret')}")
  print(f"Computed average discard critical {exp.compute_avg_stat('discard_critical')}")
  print(f"Computed average play fail {exp.compute_avg_stat('discard_critical')}")
  print(exp.game_df)
  plot_score_hist(exp, exp.game_df["progress"])

if __name__ == "__main__":
  file_path = "vrun_rulebased_2/"
  file_names = {"vrun_rulebased3p":"vrun_rulebased3p.sh.o"}
  file_names = {k:file_path+v for k,v in file_names.items()}
  master_experiments = {}
  experiments = None
  for k,v in file_names.items():
    experiments = None
    exec(open(v).read()) # Defines a list of experiments_store (eg. 2p - 5p)
    assert experiments is not None
    master_experiments[k] = experiments

  assert experiments is not None
  explain_experiment(master_experiments["vrun_rulebased3p"][0])


