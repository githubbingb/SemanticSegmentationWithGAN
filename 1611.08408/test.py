import numpy as np

a = np.array([[1,1,0],[1,0,1]])
b = np.array([[0,1,1],[0,0,1]])

a_f = a.flatten()
b_f = b.flatten()

print (a_f*b_f).sum()
