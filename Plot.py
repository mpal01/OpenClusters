

import matplotlib.pyplot as plt
import numpy as np

year = [13,14,15]
male = np.array([8,8,13])
female = np.array([1,2,10])

plt.bar(year,male,alpha=0.5,width=0.8)
plt.bar(year,male+female,alpha=0.5,width=0.8)
plt.xlabel('year')
plt.ylabel('number of books')
plt.savefig('Test',dpi=200)

