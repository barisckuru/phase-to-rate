for i in {1..10}
	do python 01S1_simulate_with_noise.py -grid_seed $i -noise_scale 0.05 &
done
