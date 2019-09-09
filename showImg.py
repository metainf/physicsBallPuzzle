import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# write your script here, we recommend the above libraries for making your animation
fig, ax = plt.subplots(1)
loaded = np.load("0C.npz")
data = loaded['ballImgArray']
plt.axis('off')
plt.tight_layout(pad=0)

N = data.shape[3]

imgax = ax.imshow(data[:,:,:,0])


def init():
    return [imgax]

def animate(i):
    global data,imgax,N,fig

    if i < N-1:  # If the animation is still running
        imgax.set_data(data[:,:,:,i+1])
        return [imgax]
    else:
        return []


# Start the animation
ani = animation.FuncAnimation(fig, animate, frames=N,
                              init_func=init, blit=True,
                              repeat=False, interval=5)
plt.show()
