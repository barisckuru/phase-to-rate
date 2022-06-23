for i in {1..30}
	do python tempotron_script_gc_merge.py -grid_seed $i -shuffling shuffled &
done
