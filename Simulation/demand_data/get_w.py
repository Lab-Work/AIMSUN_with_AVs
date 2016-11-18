import glob
import numpy as np
import matplotlib.pyplot as plt

dir = '*.txt'

files = glob.glob(dir)

all_w = []

for f in files:

	w = []
	tmp_w = []
	tmp_w.append(f)
	# get the penetration rate and 
	with open(f, 'r') as fi:

		for line in fi:
			if '_' in line:
				tmp_w.append( line.strip().split('_')[-1] )

		# remove duplicated w
		for i in range(0, len(tmp_w), 2):
			w.append(tmp_w[i])

	# append to all 
	all_w.append(w)


# write to files
fo = open('w.csv', 'w')
for i in all_w:
	fo.write( ','.join( [str(j) for j in i] ) + '\n')

fo.close()