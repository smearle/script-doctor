srun -n1 --tasks=1 --cpus-per-task=4 -t8:00:00 --mem=16G --account=torch_pr_84_tandon_advanced --pty claude "$@"
