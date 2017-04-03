import glob
import numpy as np
import matplotlib.pyplot as plt

# define color
col = {0:'r', 25:'b', 50:'g', 75:'k', 100:'y'}

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkblue', 'purple', 'hotpink']

all_w = {}    

f = open('w.csv', 'r')

for line in f:
	items = line.strip().split(':')
	dist = int(items[0])
	w = [int(i) for i in items[-1].split(',')]

	if dist not in all_w.keys():
		all_w[dist] = []

	all_w[dist].append(w)


f.close()

# plot w
fontsize=(36, 32, 28)
fig = plt.figure(figsize=(15, 8), dpi=100)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])   

key = 0
i = 0
for w in all_w[key]:
	plt.plot(w, color=colors[i], linewidth=2)
	i = i+1

plt.title('Variability of w for U(0,{0})'.format(key), fontsize=fontsize[0])
plt.xlabel('Time',fontsize=fontsize[1])   
plt.ylabel('Percent of AVs',fontsize=fontsize[1])
ax.tick_params(labelsize=fontsize[2])

plt.legend(loc = 2, fontsize=fontsize[2])
plt.xlim([0,11])
plt.ylim([0,100])

plt.savefig('w_{0}.png'.format(key), bbox_inches='tight')
plt.show()

