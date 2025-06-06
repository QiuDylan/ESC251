# %%
import numpy as np
import parameters as P
from integrators import get_integrator
from pid import PIDControl
import matplotlib.pyplot as plt

# %%
class Controller:
    def __init__(self):
        self.ctrl = PIDControl(P.kp, P.ki, P.kd, P.umax, P.sigma, P.Ts)
        
    def update(self, r, y):
        u = self.ctrl.PID(r, y)
        
        #u = 0
        return self.ctrl.saturate(u)
        
class System:
    def __init__(self):
        self.t = 0.0
        self.x = 0
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
t_history = []
y_history = []
u_history = []

T = np.linspace(0,40, P.nsteps) # implement sinusoid signal
r = 10 * np.sin(0.25 * T)
y = 0
t = 0

for i in range(P.nsteps):
    u = controller.update(r[i], y) 
    y = system.update(u) 
    t += P.Ts

    t_history.append(t)
    y_history.append(y)
    u_history.append(u)

# %%
# Plot response y due to step change in r
plt.figure()
plt.plot(t_history, y_history, label='Response (y)')
plt.plot(t_history, r, color='r', linestyle='--', label='Reference (r)')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.title('Step Response')
plt.legend()
plt.grid()
plt.show()

# Plot actuation signal
# %%
