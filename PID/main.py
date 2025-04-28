# %%
import numpy as np
import parameters as P
from integrators import get_integrator
from pid import PIDControl
import matplotlib.pyplot as plt

# %%
class Controller:
    def __init__(self):
        self.y = 0
        pass
    def update(self, r, y):
        # Compute the current error
        u = r - y
        #u = 0
        return u 
        
class System:
    def __init__(self):
        self.t = 0.0
        self.x = 10
        self.intg = get_integrator(P.Ts, self.eom)
                
    def eom(self, t, x, u):
        return (P.K * u - x) / P.tau
    
    def update(self, u):
        self.t += P.Ts
        self.x = self.intg.step(t, self.x, u)
    
        return self.x
        

# Init system and feedback controller
system = System()
controller = Controller()

# Simulate step response
t_history = [0]
y_history = [0]
u_history = [0]

r = 1
y = 0
t = 0

for i in range(P.nsteps):
    u = controller.update(r, y) 
    y = system.update(u) 
    t += P.Ts

    t_history.append(t)
    y_history.append(y)
    u_history.append(u)

# %%
# Plot response y due to step change in r
plt.figure()
plt.plot(t_history, y_history, label='Response (y)')
plt.axhline(r, color='r', linestyle='--', label='Reference (r)')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.title('Step Response')
plt.legend()
plt.grid()
plt.show()

# Plot actuation signal
# %%
