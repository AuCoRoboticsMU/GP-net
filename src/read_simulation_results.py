import os
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

from src.experiments import simulation_experiment


rootdir = 'data/experiments/'

all_files = os.listdir(rootdir)

performance = []
for f in all_files:
    logdir = Path(rootdir + f)
    data = simulation_experiment.Data(logdir)
    performance.append(data.success_rate())

performance = np.array(performance)

print("Performance: {:0.2f}".format(performance.mean(), performance.std()))
