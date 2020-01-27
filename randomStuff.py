import matplotlib.pyplot as plt
import numpy as np

inputs = np.arange(100)
outputs = np.zeros_like(inputs)
for i,score in enumerate(inputs):
  if score < 50:
    outputs[i] = score - (score - 50) * (score - 50)
  else:
    outputs[i] = score/50

fig, ax = plt.subplots()
ax.plot(inputs,outputs)
plt.show()

for i in range(10):
  print(i)
  if i > 5:
    break