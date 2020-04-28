import phyre
import numpy as np
import matplotlib.pyplot as plt

f = open("simple_agent_results2020_03_18_162202.csv", "r")

cross_results = []
cross_solved = []
within_results = []
within_solved = []

for line in f:
  lineData = line.split(',')
  if lineData[0] == "ball_cross_template":
    cross_results.append(float(lineData[2]))
    cross_solved.append(float(lineData[3]) / float(lineData[4]))
  elif lineData[0] == "ball_within_template":
    within_results.append(float(lineData[2]))
    within_solved.append(float(lineData[3]) / float(lineData[4]))

print(f.name)

if len(cross_results) > 0:
  print("Ball Cross AUCCESS  Mean:", np.mean(cross_results), "STD:", np.std(cross_results))
  print("Ball Cross Solved Mean:", np.mean(cross_solved), "STD:", np.std(cross_solved))

if len(within_results) > 0:
  print("Ball Within AUCCESS Mean:", np.mean(within_results),"STD:", np.std(within_results))
  print("Ball Within Solved Mean:", np.mean(within_solved), "STD:", np.std(within_solved))


fig, axs = plt.subplots(1, 2)
  
axs[0].boxplot(cross_results)
axs[0].set_title('Cross Template AUCCESS')

axs[1].boxplot(within_results)
axs[1].set_title('Within Template AUCCESS')

imgName = f.name.split('.')[0] + "AUCCESS.png"

plt.subplots_adjust(wspace=.3)
plt.savefig(imgName)
plt.show()

fig, axs = plt.subplots(1, 2)
  
axs[0].boxplot(cross_solved)
axs[0].set_title('Cross Template Solved')

axs[1].boxplot(within_solved)
axs[1].set_title('Within Template Solved')

imgName = f.name.split('.')[0] + "Solved.png"

plt.subplots_adjust(wspace=.3)
plt.savefig(imgName)
plt.show()


