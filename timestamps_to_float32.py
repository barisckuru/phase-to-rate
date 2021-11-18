# -*- coding: utf-8 -*-
"""
Convert time stamps to float32 dtype.
"""
import shelve, os, fnmatch, sys
import numpy as np

in_path = sys.argv[1]
out_path = sys.argv[2]

for root, dirnames, filenames in os.walk(in_path):
    # print(root)
    for filename in fnmatch.filter(filenames, '*.dat'):
        os.makedirs(out_path, exist_ok=True)

        out_shelve_path = os.path.join(out_path, filename)
        # 
        curr_file_path = os.path.join(root, '.'.join(filename.split('.')[:-1]))
        # assert os.path.isfile(curr_file_path)
        with shelve.open(curr_file_path) as sh:
            print(list(sh.keys()))
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
                            
            with shelve.open(out_shelve_path) as new_file:
                new_file['grid_spikes'] = new_shelve['grid_spikes']
                new_file['granule_spikes'] = new_shelve['granule_spikes']
                new_file['parameters'] = sh['parameters']
