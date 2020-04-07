import phyre
import numpy as np

f = open("random_agent_with_seq_results2020_03_26_100619.csv", "r")

cross_results = []

within_results = []

for line in f:
  lineData = line.split(',')
  if lineData[0] == "ball_cross_template":
    cross_results.append(float(lineData[2]))
  elif lineData[0] == "ball_within_template":
    within_results.append(float(lineData[2]))

print("Ball Cross Mean:", np.mean(cross_results), "STD:", np.std(cross_results))

print("Ball Within Mean:", np.mean(within_results),
      "STD:", np.std(within_results))
