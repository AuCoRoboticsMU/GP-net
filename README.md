# Proposing grasps for Mobile Manipulators

This repository is for training a GP-net model to propose grasps on mobile manipulators. Further, you can
analyse the resulting GP-net model in simulation.

### Installation

We recommend installation in a virtual environment, e.g. using conda or venv. The required dependencies are listed
in requirements.txt. We used python3.8.

### Training and evaluation

The data for training and pre-trained models are available 
on [zenodo](https://zenodo.org/record/7589237).
If you want to create other training data for alternative gripper configurations, generate
the data with the code available on [github](https://github.com/AuCoRoboticsMU/gpnet-data). 
Store the dataset in a directory of
your choice at `$DATASET_DIR`. All models are stored in `GP-net/data/runs/`

To train a model, run

```
python3 src/train.py --dataset $DATASET_DIR
```

To evaluate a model in simulation, you need to have download GP-net_simulation_data.zip from 
[zenodo](https://zenodo.org/record/7589237), unzip it in this repository and run

```
python3 src/experiments/simulation_experiment.py --model $MODEL_DIR
```

An example of how to read the simulation results is given in `src/read_simulation_results.py`

----------------
If you use this code, please cite

A. Konrad, J. McDonald and R. Villing, "GP-Net: Flexible Viewpoint Grasp Proposal," in 21st International Conference on Advanced Robotics (ICAR), (pp. 317-324), 2023.

along with

M. Breyer, J. J. Chung, L. Ott, S. Roland, and N. Juan, “Volumetric
grasping network: Real-time 6 dof grasp detection in clutter,” in Conference on Robot Learning, 2020
 --------
### Acknowledgements

This publication has emanated from research supported in part by Grants from Science Foundation Ireland under 
Grant numbers 18/CRT/6049 and 16/RI/3399.
The opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do 
not necessarily reflect the views of the Science Foundation Ireland.
