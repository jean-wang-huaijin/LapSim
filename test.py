import numpy as np
from LapSim import LapSim
import matplotlib.pyplot as plt

lapsim = LapSim.init_ellipse(resolution=50, steps=100) 

lapsim.lap_time()

print('Total lap time:',str(lapsim.time))
print('Average speed:',str(np.mean(lapsim.v)))
print('Track length:',str(lapsim.track_len))
plt.show()




