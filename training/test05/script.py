import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Boston.csv')
crime = pd.get_dummies(df['CRIME'], drop_first=True)
df2 = pd.concat([df, crime], axis=1)
df2 = df2.drop(['CRIME'], axis=1)

train_val, test = train_test_split(df2, test_size=0.2, random_state=0)
train_val2 = train_val.fillna(train_val.mean())
# for name in train_val2.columns:
#   if name == 'PRICE':
#     continue
#   train_val2.plot(kind='scatter', x=name, y='PRICE')
#   plt.show()

out_line1 = train_val2[ (train_val2['RM'] < 6) & (train_val2['PRICE'] > 40) ].index
out_line2 = train_val2[ (train_val2['PTRATIO'] > 18) & (train_val2['PRICE'] > 40) ].index
train_val3 = train_val2.drop([76], axis=0)

# 絞り込んだ列以外を取り除く
col = ['INDUS', 'NOX', 'RM', 'PTRATIO', 'LSTAT', 'PRICE']
train_val4 = train_val3[col]

# 相関関係の調査
# train_cor = train_val4.corr()['PRICE']
# abs_cor = train_cor.map(abs)
# print(abs_cor.sort_values(ascending=False))

# xcol = ['RM', 'LSTAT', 'PTRATIO']
# x = train_val4[xcol]
# t = train_val4[['PRICE']] #標準化をするためにデータフレーム型で格納（標準化しないならシリーズ型でも可）

def learn(x, t):
  x_train, x_val, y_train, y_val = train_test_split(x, t, test_size=0.2, random_state=0)
  # 訓練データの標準化
  sc_model_x = StandardScaler()
  sc_model_y = StandardScaler()
  sc_model_x.fit(x_train)
  sc_x_train = sc_model_x.transform(x_train)
  sc_model_y.fit(y_train)
  sc_y_train = sc_model_y.transform(y_train)
  # 学習
  model = LinearRegression()
  model.fit(sc_x_train, sc_y_train)
  # 検証用データの標準化
  sc_x_val = sc_model_x.transform(x_val)
  sc_y_val = sc_model_y.transform(y_val)
  # 訓練データと検証データの決定係数計算
  train_score = model.score(sc_x_train, sc_y_train)
  val_score = model.score(sc_x_val, sc_y_val)
  
  return train_score, val_score
  
xcol = ['RM', 'LSTAT', 'PTRATIO']
x = train_val4[xcol]
x['RM2'] = x['RM'] ** 2
x['LSTAT2'] = x['LSTAT'] ** 2
x['PTRATIO2'] = x['PTRATIO'] ** 2
x['RM * LSTAT'] = x['RM'] * x['LSTAT']
t = train_val4[['PRICE']] #標準化をするためにデータフレーム型で格納（標準化しないならシリーズ型でも可）

sc_model_x2 = StandardScaler()
sc_model_y2 = StandardScaler()
sc_model_x2.fit(x)
sc_x = sc_model_x2.transform(x)
sc_model_y2.fit(t)
sc_y = sc_model_y2.transform(t)
model = LinearRegression()
model.fit(sc_x, sc_y)


# s1, s2 = learn(x, t)
# print(s1, s2)

# 平均値・標準偏差の確認
# tmp_df = pd.DataFrame(sc_x, columns=x_train.columns)
# print(tmp_df.mean()) # 平均値
# print(tmp_df.std()) # 標準偏差

# print(model.score(sc_x_val, sc_y_val))