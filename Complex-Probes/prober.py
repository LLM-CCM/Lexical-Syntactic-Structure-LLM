from classifier import main_prog, load
import argparse
import json

import numpy as np
from senteval_tool import MLP

def main():
	print("STARTED")
	for i in range(0, 12):
		print("Running for layer: ", i+1)
		parser = argparse.ArgumentParser(description="Probing classifier")
		parser.add_argument("--labels_file", 
		                  type=str, 
		                  default=None, 
		                  help="file containing probing text and labels")
		parser.add_argument("--feats_file", 
		                  type=str,
		                  default=None, 
		                  help="file containing bert features for a probing task")
		parser.add_argument('--layer', 
		                  type=int, 
		                  default=0, 
		                  help='bert layer id to probe')
		parser.add_argument('--nhid', 
		                  type=int, 
		                  default=50, 
		                  help='hidden size of MLP')
		parser.add_argument('--dropout', 
		                  type=float, 
		                  default=0.0, 
		                  help='dropout prob. value')
		parser.add_argument('--seed', 
		                  type=int, 
		                  default=123, 
		                  help='seed value to be set manually')

		args = parser.parse_args()
		args.layer = i
		print(args)
		train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes = load(args)
		main_prog(args, train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes)
	print("DONE")

if __name__ == "__main__":
    main()