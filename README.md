# Phase to Rate Coding
This is the code underlying a research project where we investigate how spatial phase information in grid cells is processed in a biophysically
realistic model of the dentate gyrus. The main purpose of this repository is to make the code we wrote for the project available and reproduce our findings.

# Setup
- Open a terminal and clone the repository `git clone https://github.com/barisckuru/phase-to-rate.git`.
- Create a fresh conda environment `conda create -n phasetorate python=3.9`.
- Activate your new environment `conda activate phasetorate`.
- Install git if needed `conda install git` (pip runs into problems without git in the environment).
- Inside the cloned repository pip install phase-to-rate `pip install -e .`
- Our dentate gyrus simulation is based on NEURON, which cannot be pip installed on Windows.
You therefore need to install it NEURON manually on Windows. [Get the precompiled Windows installer here](https://www.neuron.yale.edu/neuron/download)
Use the installer to install neuron in `C:\nrn`. Once installed you need to find a way to add `C:\nrn\lib\python` to your Python path. [Try something like this](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages). Sometimes you also need to manually set the environmental variable `NEURONHOME` to `C:\nrn` but most of the time the installer should take care of that. To check if it worked, start `python` and try `from neuron import h, gui`. If this runs without error NEURON is installed.
- Try `import phase_to_rate` to check that the setup worked.
- Run `python 01_simulate.py` to start generating some raw data.
  
If you encounter problems with creating the setup that relate to pydentate you can look for a solution in the [pydentate repo](https://github.com/danielmk/pydentate). There are some known issues that relate to installing NEURON on windows. You can contact danielmuellermsc@gmail.com for technical questions relating to this repository and the setup.

# Reproducibility
The easiest way to reproduce the findings is with the original raw data. Since it is 70GB large we unfortunately cannot ship it with the GitHub repo. If you want to get the raw data, contact the corresponding author Oliver Branganza (oliver.braganza@ukbonn.de). With the original data in your phase-to-rate directory you can simply run the scripts `figure1.py` to `figure5.py` interactively to reproduce the figures. If you want to generate and analyze the raw data yourself or just want to see the code behind it, you will need to use the following scripts in order:
- 01_simulate.py
    - This script does the heavy lifting. It simulates the grid cell spikes and feeds them into pydentate. The script saves shelve files that contain the grid and granule cell spike times as well as the important parameters of the simulation.
- 02_pearsonr.py
    - This is an analysis script that loads grid as well granule cell spikes generated in 01_simulate.py and calculates input/output Pearson correlations for different trajectories. It creates two excel sheets with the correlations values.
- 03_information.py
    - This analysis script loads grid and granule cell spikes and calculates the Skaggs spatial information measure for a set of trajectories. Output are two excel files.
- 04_perceptron.py
    - This script performs the perceptron training for different pairs of trajectories. It creates pickle files containing the learning curves.
- 05_tempotron.py
    - This script loads granule spikes and trains a tempotron to distinguish two trajectories. Aggregate results are saved in a sqlite3 database and learning curve arrays are saved as .npy files.
- 06_simulate_ca3.py
    - Loads the granule spikes and feeds them into a model of CA3. Pickles the results.

These scripts depend on modules in `phase_to_rate`. A brief explanation on those:
- figure_functions.py
    - Utility functions relating to plotting results in the figure scripts.
- grid_model.py
    - Functions that simulate the spikes from phase precessing grid cells during straight behavioral trajectories.
- information_measure.py
    - Functions relating to Skaggs information measure.
- neural_coding.py
    - Functions to convert spikes to binned phase and rate codes.
- perceptron.py
    - Functions to train the perceptron with pytorch.
- pydenate_integrate.py
    - Functions to simulate pydentate with grid cell input.

# Authors
Barış Can Kuru - [Institute of Experimental Epileptology and Cognition Research](https://eecr-bonn.de/)

Daniel Müller-Komorowska - [Neural Coding and Brain Computing Unit, Okinawa Institute of Science and Technology Graduate University, Okinawa, Japan](https://groups.oist.jp/ncbc)

Oliver Braganza - [Institute of Experimental Epileptology and Cognition Research](https://beck-group.ieecr-bonn.de/member/dr-oliver-braganza/)