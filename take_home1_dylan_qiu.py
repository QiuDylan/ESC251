# %%
# ESC 251 Take Home Exam 1
# Contributor: Dylan Qiu, The Cooper Union
import numpy as np
from control.matlab import * 
import matplotlib.pyplot as plt

# %%
#Constants 
Ra = 0.02 
Rb = 2.0
Rc = 2.2
Rd = 0.2 
Re = 0.02
C1 = 8700
C2 = 6200
C3 = 6600 
C4 = 2e4

#State space matrices
A = np.array([[-1/(C1*Ra)-1/(C1*Rb) , 1/(C1*Rb),0,0],
             [1/(C2*Rb), -1/(C2*Rb)-1/(C2*Rc),1/(C2 *Rc), 0],
             [0, 1/(C3*Rc), -1/(C3*Rc)-1/(C3*Rd),1/(C3 *Rd)],
             [0, 0,  1/(C4 *Rd), -1/(C4*Rd)-1/(C4*Re)]])
B = np.array([[0], [0], [0], [1/(C4*Re)]])
C = np.eye(4)
D = np.zeros((4,1))
print(A)
sys = ss(A, B, C, D)
# %%
#Step response
t = np.linspace(0, 10e4, 1000)
y,t = step(sys, t)
y = np.squeeze(y)
plt.plot(t, y[:,0], label=r'Drywall T1')
plt.plot(t, y[:,1], label=r'Fiberglass T2')
plt.plot(t, y[:,2], label=r'Plywood T3')
plt.plot(t, y[:,3], label=r'Concrete Block T4')
plt.ylim(0,5e-3)
plt.grid(True)
plt.xlabel('Time') 
plt.ylabel('Temperature deg(c)')
plt.title('Step response')
plt.legend()
plt.show()
# %%
# Part E
Ti = np.linspace(20,20, 1000)
T1 = T2 = T3 = T4 = 15
T = np.linspace(0,3600, 1000)
To = 5 - 15 * T / 3600
u = To - Ti 
yout, t , xout = lsim(sys, U = u, T=T, X0 = [[T1 - 20], [T2 - 20], [T3 - 20], [T4- 20]])
T1 = yout[:,0] + Ti
T2 = yout[:,1] + Ti
T3 = yout[:,2] + Ti
T4 = yout[:,3] + Ti
plt.plot(t, T1, label=r'Drywall T1')
plt.plot(t, T2, label=r'Fiberglass T2')
plt.plot(t, T3, label=r'Plywood T3')
plt.plot(t, T4, label=r'Concrete Block T4')
plt.plot(t, To, label=r'External Temp To', linestyle = 'dashed')
plt.plot(t, Ti, label=r'Internal Temp Ti', linestyle = 'dashed')

plt.grid(True)
plt.xlabel('Time') 
plt.ylabel('Temperature deg(c)')
plt.title('Temperature Evolution due to Input')
plt.legend()
plt.show()
# %%

#Pole plot
p = poles(sys)
print(p)
plt.plot(np.real(p)[0], np.imag(p)[0], 'x')
plt.plot(np.real(p)[1], np.imag(p)[1], 'x')
plt.plot(np.real(p)[2], np.imag(p)[2], 'x')
plt.plot(np.real(p)[3], np.imag(p)[2], 'x')

plt.grid(True)
plt.axhline(0,color = 'black')
plt.axvline(0,color = 'black')
plt.xlabel('real')
plt.ylabel('imaginary')
plt.title('pole plot')
plt.show()
# %%
