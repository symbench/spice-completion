This repo contains code for exploring auto completion of spice netlists.

## Quick Start
First install the dependencies (as shown below). Then follow the instructions for whatever technique you would like to try. (Use Python 3)
```bash
pip install -r requirements.txt
```

### Graph (Attention) Neural Network
```bash
python train.py path/to/netlist.net
```

Since it is still far from complete, it is probably best to run in the interpreter so you can explore the trained model, etc:
```bash
python -i train.py path/to/netlist.net
```

### K-Nearest Neighbors
Using KNN seems to be the closest analog to n-grams from NLP in the graph context. This is a simple approach where we are simply trying to use the component type with the most similar neighborhood that we have seen in the training set. Each point is represented as a list of the pins where each pin is represented as a sorted list of the other elements at the node. Points are compared using hamming (edit) distance.

The approach is still pretty simple as it just uses the closest neighbor currently (K=1).

```
python train_knn.py path/to/netlist path/to/netlist2 ...
```

```
python predict_knn.py model_path path/to/netlist path/to/netlist2 ...
```
