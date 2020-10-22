from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice import BasicElement
from PySpice.Spice.Netlist import Pin
from knn_utils import ComponentPoint, KNNModel

import sys
import pickle

def get_component_points(source):
    return [ ComponentPoint(element) for element in circuit.elements ]

model_path = sys.argv[1]
model = pickle.load(open(model_path, 'rb'))

filenames = sys.argv[2:]
correct = 0.
total = 0.
for filename in filenames:
    with open(filename, 'rb') as f:
        source = f.read().decode('utf-8', 'ignore')
        try:
            parser = SpiceParser(source=source)
            circuit = parser.build_circuit()
            for element in circuit.elements:
                point = ComponentPoint(element)
                pred = model.closest(point)
                if pred.label == point.label:
                    correct += 1
                total += 1
        except:
            pass

print('Accuracy:', correct/total)
