import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

#path = './data.csv'
#particle_count = str(sys.argv[1])
#path = 'test_particles' + particle_count + '.csv'

#data = pd.read_csv(path, encoding='utf8')

double_time = [0.00578061, 0.00600652, 0.00920814, 0.0407606, 0.304905, 2.88041]
#single_time = [2.85146, 1.42738, 1.40818, 1.41135, 1.40663, 1.39792, 1.38382]
size = [1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000]

# Create figure
fig = plt.figure()

# Reference to axis
ax = fig.add_subplot()

# Add scatters 
#ax.scatter(data['threads'], data['gpu'], label='GPU')
#ax.scatter(data['threads'], data['cpu'], label='CPU')
#ax.set_xscale('log')

ax.scatter(np.log(size), double_time, 50, label='GPU')

plt.title("GPU time")

# Plot title and axis labels
#plt.title("Comparison between GPU and CPU for particle simulation with {} particles.".format(data['particles'][0]))
plt.ylabel('GPU Simulation Time (s)')
plt.xlabel('Iterations')

# Legend
plt.legend(loc='upper left')

plt.show()
plt.savefig("name.png")