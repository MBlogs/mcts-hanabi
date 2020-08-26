# MuteAgent
python3 run_experiment.py --num_episodes 250 --agent MuteAgent --agents OuterAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent MuteAgent --agents IGGIAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent MuteAgent --agents PiersAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent MuteAgent --agents VanDenBerghAgent >> mix_rulebased.txt
# LegalRandomAgent
python3 run_experiment.py --num_episodes 250 --agent LegalRandomAgent --agents OuterAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent LegalRandomAgent --agents IGGIAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent LegalRandomAgent --agents PiersAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent LegalRandomAgent --agents VanDenBerghAgent >> mix_rulebased.txt
# FlawedAgent
python3 run_experiment.py --num_episodes 250 --agent FlawedAgent --agents OuterAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent FlawedAgent --agents IGGIAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent FlawedAgent --agents PiersAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent FlawedAgent --agents VanDenBerghAgent >> mix_rulebased.txt
# InnerAgent
python3 run_experiment.py --num_episodes 250 --agent InnerAgent --agents OuterAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent InnerAgent --agents IGGIAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent InnerAgent --agents PiersAgent >> mix_rulebased.txt
python3 run_experiment.py --num_episodes 250 --agent InnerAgent --agents VanDenBerghAgent >> mix_rulebased.txt