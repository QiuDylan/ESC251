# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
from control.matlab import * # MATLAB-like control toolbox functionality

# %%
# Close all figures
plt.close('all')
# Set up bulb-temperature model
# Input: electric energy, Output: Temperature (above ambient)
a, b = (-.0067, .67)
sys = ss(a,b,1,0)

# (1) Turn on lamp and measure bulb temperature
t1 = np.linspace(0, 1000, 100) # integrate model for T sec.
y1, t1 = step(sys, t1)
fig, ax = plt.subplots()
ax.plot(t1, y1, 'r')

# (2) Turn off lamp and measure bulb temperature
t2 = np.linspace(t1[-1], t1[-1] + 1500, 100)
y2, t2 = initial(sys, t2, y1[-1])
ax.plot(t2, y2, 'b')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Temperature above ambient [K]'),
ax.legend(('bulb ON (warming up)','bulb OFF (cooling down)'))
#axis([0 2500 0 110])
plt.show()

# %%
