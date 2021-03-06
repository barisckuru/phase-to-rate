#!/usr/bin/env bash
python spike_generation_script_large_scale.py -gs 0 -ps 2000 3000 1 -t 75 -s "non-shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 11000 12000 1 -t 73 -s "non-shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 101000 102000 1 -t 70 -s "non-shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 1001000 1002000 1 -t 65 -s "non-shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 10001000 10002000 1 -t 60 -s "non-shuffled" -n "full" &

python spike_generation_script_large_scale.py -gs 0 -ps 1000 2000 1 -t 75 -s "shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 10000 11000 1 -t 73 -s "shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 100000 101000 1 -t 70 -s "shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 1000000 1001000 1 -t 65 -s "shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 10000000 10001000 1 -t 60 -s "shuffled" -n "full" &

python spike_generation_script_large_scale.py -gs 0 -ps 2000 3000 1 -t 75 -s "shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 11000 12000 1 -t 73 -s "shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 101000 102000 1 -t 70 -s "shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 1001000 1002000 1 -t 65 -s "shuffled" -n "full" &
python spike_generation_script_large_scale.py -gs 0 -ps 10001000 10002000 1 -t 60 -s "shuffled" -n "full" &

python spike_generation_script_large_scale.py -gs 0 -ps 1000 2000 1 -t 75 -s "non-shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 10000 11000 1 -t 73 -s "non-shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 100000 101000 1 -t 70 -s "non-shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 1000000 1001000 1 -t 65 -s "non-shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 10000000 10001000 1 -t 60 -s "non-shuffled" -n "no-feedback" &

python spike_generation_script_large_scale.py -gs 0 -ps 2000 3000 1 -t 75 -s "non-shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 11000 12000 1 -t 73 -s "non-shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 101000 102000 1 -t 70 -s "non-shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 1001000 1002000 1 -t 65 -s "non-shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 10001000 10002000 1 -t 60 -s "non-shuffled" -n "no-feedback" &

python spike_generation_script_large_scale.py -gs 0 -ps 1000 2000 1 -t 75 -s "shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 10000 11000 1 -t 73 -s "shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 100000 101000 1 -t 70 -s "shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 1000000 1001000 1 -t 65 -s "shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 10000000 10001000 1 -t 60 -s "shuffled" -n "no-feedback" &

python spike_generation_script_large_scale.py -gs 0 -ps 2000 3000 1 -t 75 -s "shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 11000 12000 1 -t 73 -s "shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 101000 102000 1 -t 70 -s "shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 1001000 1002000 1 -t 65 -s "shuffled" -n "no-feedback" &
python spike_generation_script_large_scale.py -gs 0 -ps 10001000 10002000 1 -t 60 -s "shuffled" -n "no-feedback" &