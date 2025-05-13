sync_data_210:
	rsync -arv ./data/js_sols 210:~/script-doctor/data && \
	rsync -arv 210:~/script-doctor/data ./