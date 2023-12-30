from cycler import cycler 
import numpy as np 
import matplotlib.pyplot as plt 


plt.rc('figure', figsize=(8, 7.5))

A = np.random.randn(100)
B = np.random.randn(100)

plt.plot(A, 'r-o', label='Array A')
plt.plot(B, 'g--x', label='Array B')

plt.legend()
plt.show()


