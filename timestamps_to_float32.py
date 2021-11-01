# -*- coding: utf-8 -*-
"""
Convert time stamps to float32 dtype.
"""

path = r'E:\Dropbox\Dropbox\data\network_full\seed_0\grid-seed_trajectories_poisson-seeds_duration_shuffling_tuning_0_[60.0]_[10000000, 10000200, 1]_2000_non-shuffled_full'

sh = shelve.open(path)

new_shelve = {'grid_spikes': 
              {traj: {ps:[] for ps in sh['grid_spikes'][traj].keys()}
               for traj in sh['grid_spikes'].keys()},
              'granule_spikes': 
              {traj: {ps:[] for ps in sh['granule_spikes'][traj].keys()}
               for traj in sh['granule_spikes'].keys()}}

for ct in ['grid_spikes', 'granule_spikes']:
    for traj in sh[ct].keys():
        for ps in sh[ct][traj].keys():
            for idx, ts in enumerate(sh[ct][traj][ps]):
                new_shelve[ct][traj][ps].append(sh[ct][traj][ps][idx].astype(np.float32))
                
new_file = shelve.open(r'E:\Dropbox\Dropbox\data\network_full\seed_100\smaller_float')

new_file['grid_spikes'] = new_shelve['grid_spikes']
new_file['granule_spikes'] = new_shelve['granule_spikes']
new_file['parameters'] = sh['parameters']


new_file.close()
