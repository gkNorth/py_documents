import pandas as pd

df = pd.read_csv('KvsT.csv')

xcol = ['身長']
f = df[xcol]
t = df['派閥']

import matplotlib.pyplot as plt
x = [4,9,14]
y = [10,19,25]
plt.plot(f)
plt.show()