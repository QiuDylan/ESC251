# DC motor 
K = 4.9 # rad/s/V 
tau = 0.085 # 1/s
umax = 12 # V
udead = 0.8 # V, deadband

# Simulation
rstep = 20
Ts = 3e-3 # s
nsteps = 135

# PID controller
emax = 0.5*K*umax
zeta = 0.7 
kp = 0.95*umax/emax
wn = (1 + K*kp)/(2*tau*zeta)
ki = wn**2 * tau / K
kd = 0
sigma = 0.01
