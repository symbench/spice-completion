import numpy as np
from knn_utils import ComponentPoint, KNNModel

import sys
import pickle

def deep_equals(l1, l2):
    are_both_lists = all((type(l) is list for l in (l1, l2)))
    if not are_both_lists:
        return l1 == l2

    if len(l1) == len(l2):
        for (i1, i2) in zip(l1, l2):
            if not deep_equals(i1, i2):
                return False

        return True
    return False

model_path = sys.argv[1]
model = pickle.load(open(model_path, 'rb'))
# TODO: Check out the number of collisions
visited_points = []
for point in model.points:
    is_already_visited = any((deep_equals(point.data, d) for d in visited_points))
    if is_already_visited:
        continue

    visited_points.append(point.data)
    duplicates = model.matches(point)
    labels = [ pt.label for pt in duplicates ]
    labels, counts = np.unique(labels, return_counts=True)
    if len(labels) > 1:
        print('duplicate labels for', point.data)
        for (label, count) in zip(labels, counts):
            print(f'{label} {count}')
