from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice import BasicElement
from PySpice.Spice.Netlist import Pin
from knn_utils import ComponentPoint, KNNModel

import sys
import pickle

def get_component_points(source):
    parser = SpiceParser(source=source)
    circuit = parser.build_circuit()
    return [ ComponentPoint(element) for element in circuit.elements ]

model = KNNModel()
filenames = sys.argv[1:]
for filename in filenames:
    with open(filename, 'rb') as f:
        try:
            source = f.read().decode('utf-8', 'ignore')
            model.points.extend(get_component_points(source))
        except:
            pass

print('model has', len(model.points), 'points')
pickle.dump(model, open('knn-model.pkl', 'wb'))
