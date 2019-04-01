#!/usr/bin/env python

import numpy as np

dat = np.genfromtxt('magic04_label.data', delimiter=',')
glen = 0
for i in range(len(dat)):
	if dat[i][10] == 1:
		glen = i
		break
g = dat[:glen, :10]
h = dat[glen:, :10]
print h