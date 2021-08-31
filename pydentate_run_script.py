from pydentate import net_tunedrev, neuron_tools
from grid_model import grid_simulate
import shelve
import time
import numpy as np
import scipy.stats as stats

start = time.time()


"""Setup"""
neuron_tools.load_compiled_mechanisms(path='/home/baris/pydentate/mechs/x86_64/libnrnmech.so')

"""Parameters"""
grid_seeds = [1]
poisson_seeds = [100, 101]
trajectories = [75, 73]  # In cm
shuffled = ["shuffled", "non-shuffled"]
poisson_reseeding = True  # Controls seeding between trajectories
speed_cm = 20
dur_ms = 1000
rate_scale = 5
n_grid = 200
pp_weight = 9e-4
input_scale = 1000
network_type = 'tuned'

print('grid', grid_seeds, 'poiss', poisson_seeds, 'traj', trajectories, 'dur', dur_ms)

for grid_seed in grid_seeds:
    for poisson_seed in poisson_seeds:
        storage = {'grid_spikes': {},
                    'granule_spikes': {}}
        for shuffling in shuffled:
            grid_spikes, _ = grid_simulate(trajs=trajectories,
                                        dur_ms=dur_ms,
                                        grid_seed=grid_seed,
                                        poiss_seeds=poisson_seed,
                                        shuffle=shuffling,
                                        diff_seed=poisson_reseeding,
                                        n_grid=n_grid,
                                        speed_cm=speed_cm,
                                        rate_scale=rate_scale)
            granule_spikes = {}
            granule_spike_list = []
            for curr_grid_spikes in grid_spikes[poisson_seed]:
                np.random.seed(grid_seed)

                # Randomly choose target cells for the PP lines
                gauss_gc = stats.norm(loc=1000, scale=input_scale)
                gauss_bc = stats.norm(loc=12, scale=(input_scale/2000.0)*24)
                pdf_gc = gauss_gc.pdf(np.arange(2000))
                pdf_gc = pdf_gc/pdf_gc.sum()
                pdf_bc = gauss_bc.pdf(np.arange(24))
                pdf_bc = pdf_bc/pdf_bc.sum()
                GC_indices = np.arange(2000)
                start_idc = np.random.randint(0, 1999, size=400)
            
                PP_to_GCs = []
                for x in start_idc:
                    curr_idc = np.concatenate((GC_indices[x:2000], GC_indices[0:x]))
                    PP_to_GCs.append(np.random.choice(curr_idc, size=100, replace=False,
                                                      p=pdf_gc))
            
                PP_to_GCs = np.array(PP_to_GCs)
                PP_to_GCs = PP_to_GCs[0:24]
            
                BC_indices = np.arange(24)
                start_idc = np.array(((start_idc/2000.0)*24), dtype=int)
            
                PP_to_BCs = []
                for x in start_idc:
                    curr_idc = np.concatenate((BC_indices[x:24], BC_indices[0:x]))
                    PP_to_BCs.append(np.random.choice(curr_idc, size=1, replace=False,
                                                      p=pdf_bc))
            
                PP_to_BCs = np.array(PP_to_BCs)
                PP_to_BCs = PP_to_BCs[0:24]
            
                nw = net_tunedrev.TunedNetwork(None, curr_grid_spikes,
                                  PP_to_GCs,
                                  PP_to_BCs,
                                  pp_weight=pp_weight)

                # Handle the different cases of inhibition
                if network_type == 'no-feedback':
                    # Set GC to MC weight to 0
                    for syn in nw.populations[0].connections[24].netcons: syn[0].weight[0] = 0.0
                    # Set GC to BC weight to 0
                    for syn in nw.populations[0].connections[25].synapses: syn[0].weight[0] = 0.0
                elif network_type == 'no-feedforward':
                    # Set PP to BC weight to 0
                    for pp_conns in nw.populations[2].connections[0:24]:
                        for syn in pp_conns.netcons: syn.weight[0] = 0.0
                elif network_type == 'disinhibited':
                    # Set GC to MC weight to 0
                    for syn in nw.populations[0].connections[24].netcons: syn[0].weight[0] = 0.0
                    # Set GC to BC weight to 0
                    for syn in nw.populations[0].connections[25].synapses: syn[0].weight[0] = 0.0
                    # Set PP to BC weight to 0
                    for pp_conns in nw.populations[2].connections[0:24]:
                        for syn in pp_conns.netcons: syn.weight[0] = 0.0
                elif network_type != 'tuned':
                    raise ValueError("network_type must be 'tuned', 'no-feedback', 'no-feedforward' or 'disinhibited'")
                
                neuron_tools.run_neuron_simulator()
                
                granule_spike_list.append(nw.populations[0].get_timestamps())

            granule_spikes[poisson_seed] = granule_spike_list
            storage['grid_spikes'][shuffling] = grid_spikes
            storage['granule_spikes'][shuffling] = granule_spikes
            
        output_path = f'{grid_seed}_{poisson_seed}_{poisson_reseeding}'
        # save_storage TODO shelve


# "grid_seed_poisson_seed_reseeding-True"


stop = time.time()
time_sec = stop-start
time_min = time_sec/60
time_hour = time_min/60
print('time, ', time_sec, ' sec, ', time_min, ' min, ', time_hour, 'hour  ')