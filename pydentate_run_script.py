from pydentate import net_tunedrev, neuron_tools
from grid_model import grid_simulate
import shelve

"""Setup"""
neuron_tools.load_compiled_mechanisms(path='precompiled')

"""Parameters"""
grid_seeds = [1]
poisson_seeds = [100]
trajectories = [75,74]  # In cm
shuffled = ["shuffled", "non-shuffled"]
poisson_reseeding = True  # Controls seeding between trajectories
speed_cm = 20
dur_ms = 2000
rate_scale = 1
n_grid = 200
pp_weight = 9e-4
network_type = 'tuned'  # TODO Ignore for now implement more models later


for grid_seed in grid_seeds:
    for poisson_seed in poisson_seeds:
        storage = {'grid_spikes': {},
                    'granule_spikes': {}}
        for shuffling in shuffled:
            grid_spikes = grid_simulate(trajs=trajectories,
                                        dur_ms=dur_ms,
                                        grid_seed=grid_seed,
                                        poiss_seeds=poisson_seed,
                                        shuffle=shuffling,
                                        n_grid=n_grid,
                                        speed_cm=speed_cm,
                                        rate_scale=rate_scale)
            granule_spikes = {}
            granule_spike_list = []
            for curr_grid_spikes in grid_spikes[poisson_seed]:
                nw = net_tunedrev()  # TODO Pass parameters
                # TODO change weights
                # TODO Future feature, change inhibitory connections
                # if network_type == 'no-feedback':
                # nw.populations[0].connections[k].weight = 0
                granule_spike_list.append(nw.populations[0].get_timestamps())

            granule_spikes[poisson_seed] = granule_spike_list
            storage['grid_spikes'][shuffling] = grid_spikes
            storage['granule_spikes'][shuffling] = granule_spikes
            
        output_path = f'{grid_seed}_{poisson_seed}_{poisson_reseeding}'
        # save_storage TODO shelve


"grid_seed_poisson_seed_reseeding-True"


