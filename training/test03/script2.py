import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('ex3.csv')
df2 = df.fillna(df.median())

# df2.plot(kind='scatter', x=f'x2', y='target')
# plt.show()

no = df2[ (df2['target'] > 100) & (df2['x2'] < -2 )].index
df3 = df2.drop(no, axis=0)

x = df3.loc[:,:'x3']
t = df3['target']

x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

# print(x_train)

# print(df2)