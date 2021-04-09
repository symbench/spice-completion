from spice_completion.datasets import helpers as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
args = parser.parse_args()

netlists = ( (f, open(f, 'rb').read().decode('utf-8', 'ignore')) for f in args.files )
valid_files = [ f for (f, src) in netlists if h.is_valid_netlist(src)]
for f in valid_files:
    print(f)
