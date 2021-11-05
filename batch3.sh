#!/usr/bin/env bash
python spike_generation_script_large_scale.py -gs 0 -ps 1000 1500 1 -t 75 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 1500 2000 1 -t 75 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 2000 2500 1 -t 75 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 2500 3000 1 -t 75 -s "non-shuffled" -n "no-feedforward" &

python spike_generation_script_large_scale.py -gs 0 -ps 10000 10500 1 -t 73 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 10500 11000 1 -t 73 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 11000 11500 1 -t 73 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 11500 12000 1 -t 73 -s "non-shuffled" -n "no-feedforward" &

python spike_generation_script_large_scale.py -gs 0 -ps 100000 100500 1 -t 70 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 100500 101000 1 -t 70 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 101000 101500 1 -t 70 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 101500 102000 1 -t 70 -s "non-shuffled" -n "no-feedforward" &

python spike_generation_script_large_scale.py -gs 0 -ps 1000000 1000500 1 -t 65 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 1000500 1001000 1 -t 65 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 1001000 1001500 1 -t 65 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 1001500 1002000 1 -t 65 -s "non-shuffled" -n "no-feedforward" &

python spike_generation_script_large_scale.py -gs 0 -ps 10000000 10000500 1 -t 60 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 10000500 10001000 1 -t 60 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 10001000 10001500 1 -t 60 -s "non-shuffled" -n "no-feedforward" &
python spike_generation_script_large_scale.py -gs 0 -ps 10001500 10002000 1 -t 60 -s "non-shuffled" -n "no-feedforward" &

