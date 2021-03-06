import numpy as np 
import matplotlib.pyplot as plt 

greyhounds = 500
labs = 500

#setting dogs height with standard normal distribution
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height 	= 24 + 4 * np.random.randn(labs)

#draw histogram
plt.hist([grey_height, lab_height], stacked = True, color = ['r','b'])
plt.show()

