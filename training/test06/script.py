import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('Bank.csv')
# ダミー変数化
job = pd.get_dummies(df['job'], drop_first=True)
marital = pd.get_dummies(df['marital'], drop_first=True)
education = pd.get_dummies(df['education'], drop_first=True)
default = pd.get_dummies(df['default'], drop_first=True, prefix='default')
housing = pd.get_dummies(df['housing'], drop_first=True, prefix='housing')
loan = pd.get_dummies(df['loan'], drop_first=True, prefix='loan')
df2 = pd.concat([job, df], axis=1)
df2 = pd.concat([marital, df2], axis=1)
df2 = pd.concat([education, df2], axis=1)
df2 = pd.concat([default, df2], axis=1)
df2 = pd.concat([housing, df2], axis=1)
df2 = pd.concat([loan, df2], axis=1)
df2 = df2.drop(['job', 'marital', 'education', 'default', 'housing', 'loan'], axis=1)
df2 = df2.drop(['id', 'day', 'month', 'duration', 'contact'], axis=1)
print(df2)
# データ分割
train_val, test = train_test_split(df2, test_size=0.2, random_state=0)
x = df2.loc[:,:'previous']
t = df['y']
def learn(x, t, depth=3):
  x_train, x_val, y_train, y_val = train_test_split(x, t, test_size=0.2, random_state=0)
  model = tree.DecisionTreeClassifier(max_depth=depth,random_state=0,class_weight='balanced')
  model.fit(x_train, y_train)
  
  score = model.score(x_train, y_train)
  score2 = model.score(x_val, y_val)
  return round(score, 3), round(score2, 3), model

for j in range(1,15):
  train_score, test_score, model = learn(x, t, depth=j)
  print(f'深さ{j}:train正解率{train_score} test正解率{test_score}')

# 欠損値
# duration = round(train_val['duration'].isnull().sum() / len(train_val) * 100, 1)
# print(f'欠損値の割合 : {duration}%')
