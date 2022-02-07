import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('Survived.csv')
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# x = df[col]
# t = df['Survived']

def learn(x, t, depth=3):
  x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)
  model = tree.DecisionTreeClassifier(max_depth=depth,random_state=0,class_weight='balanced')
  model.fit(x_train, y_train)
  
  score = model.score(x_train, y_train)
  score2 = model.score(x_test, y_test)
  return round(score, 3), round(score2, 3), model

# train_score, test_score, model = learn(x, t, depth=5)

df2 = pd.read_csv('Survived.csv')
# df2.groupby('Survived').mean()['Age']

pt = pd.pivot_table(df2, index='Survived', columns='Pclass', values='Age')
# print(pt)

is_null = df2['Age'].isnull()
for k in range(1,4):
  for l in range(0,2):
    df2.loc[ (df2['Survived']==l) & (df2['Pclass']==k) & (is_null), 'Age' ] = pt.at[l,k]
    
col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']
x = df2[col]
t = df2['Survived']
    
# for j in range(1,15):
#   train_score, test_score, model = learn(x, t, depth=j)
#   print(f'深さ{j}:train正解率{train_score} test正解率{test_score}')

# print(df2.loc[ (df2['Survived']==1) & (df2['Pclass']==3) & (is_null), 'Age' ])

# sex = df2.groupby('Sex').mean()
# sex['Survived'].plot(kind='bar')
# plt.show()

male = pd.get_dummies(df2['Sex'], drop_first=True)
embarked = pd.get_dummies(df2['Embarked'], drop_first=True)

x_tmp = pd.concat([x, male], axis=1)
x_new = x_tmp.drop('Sex', axis=1)

train_score, test_score, model = learn(x_new, t, depth=6)

# print(f'train正解率{train_score} test正解率{test_score}')

# with open('Survived.pkl', 'wb') as f:
#   pickle.dump(model, f)

importances = pd.DataFrame(model.feature_importances_, index=x_new.columns)

print(importances)