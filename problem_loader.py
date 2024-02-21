import numpy as np

def load_problem(filename):
	A, B, C, D, E = [], [], [], [], []

	with open(filename) as f:
		comment_count = 0
		lines = f.readlines()
		for line in lines:
			if line[0] == "%":
				comment_count += 1
				continue
			
		f.close()
	