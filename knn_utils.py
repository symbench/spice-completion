import numpy as np
from scipy.spatial.distance import hamming

class ComponentPoint():
    def __init__(self, element):
        self.label = element.name[0]
        self.data = [self.pin_signature(pin) for pin in element.pins]

    def pin_signature(self, pin):
        pins = ( p for p in list(pin.node.pins) if p is not pin )
        elements = ( pin.element for pin in pins )
        return [ e.name[0] for e in elements ]

    def get_data(self, index):
        try:
            return self.data[index]
        except:
            return []

    def distance(self, otherPoint):
        max_len = max(len(otherPoint.data), len(self.data))
        dist = 0.
        for i in range(max_len):
            l1, l2 = pad_to_match(self.get_data(i), otherPoint.get_data(i))
            l1.sort()
            l2.sort()
            dist += hamming(l1, l2)/max_len
        return dist

    def __str__(self):
        return f'{self.data} ({self.label})'

def pad_to_match(l1, l2):
    max_len = max(len(l1), len(l2))
    return pad(l1, max_len), pad(l2, max_len)

def pad(l, length):
    l = l[:]
    while len(l) < length:
        l.append('_')
    return l

class KNNModel():
    def __init__(self, points=[]):
        self.points = points

    def closest(self, new_point):
        dists = np.array([new_point.distance(point) for point in self.points])
        #print('distances:')
        #for (dist, pt) in zip(dists, self.points):
            #print(pt, dist)
        return self.points[dists.argmin()]

    def predict(self, new_point):
        return self.closest(new_point).label
