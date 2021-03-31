import math
import numpy as np

from spice_completion.datasets import PrototypeLinkDataset
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
args = parser.parse_args()

dataset = PrototypeLinkDataset(args.files)
np.save('mean.npy', dataset.mean)
np.save('stddev.npy', dataset.std)
