import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cinema.csv')
df2 = df.fillna(df.mean())

# 散布図での外れ値確認
# for name in df.columns:
#   if name == 'cinema_id' or name == 'sales':
#     continue
#   df2.plot(kind='scatter', x=name, y='sales')
# plt.show()

no = df2[(df2['SNS2'] > 1000) & (df2['sales'] < 8500)].index
df3 = df2.drop(no, axis=0)

x = df3.loc[ : , 'SNS1':'original']
t = df3['sales']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train, y_train)

# new = [[150, 700, 300, 0]]
# print(model.predict(new))

# MAE : 平均絶対誤差
# from sklearn.metrics import mean_absolute_error
# pred = model.predict(x_test)
# calc = mean_absolute_error(y_pred=pred, y_true=y_test)
# print(calc)

# 決定係数
R2 = model.score(x_test, y_test)
# print(R2)

# import pickle
# with open('cinema.pkl', 'wb') as f:
#   pickle.dump(model, f)

tmp = pd.DataFrame(model.coef_)
tmp.index = x_train.columns
print(tmp)
print(model.intercept_)