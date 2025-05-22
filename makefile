sync_data_210:
	rsync -arv 210:~/script-doctor/data ./

sync_data_greene:
	rsync -arv greene:/scratch/se2161/script-doctor/data ./

sync_data:
	# rsync -arv 210:~/script-doctor/data ./
	# rsync -arv greene:/scratch/se2161/script-doctor/data ./
	rsync -arv --exclude="data/simplified_games" --exclude="data/min_games" --exclude="data/pretty_trees" --exclude="data/game_trees" --exclude="*frames" greene:/scratch/se2161/script-doctor/data ./

sync_rl_greene:
	rsync -arv greene:/scratch/se2161/script-doctor/rl_logs ./
	rsync -arv greene:/scratch/se2161/script-doctor/submitit_logs ./