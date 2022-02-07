import pandas as pd
df = pd.read_csv('iris.csv')
# df2 = df.dropna(how='any', axis=0)
colmean = df.mean()
df2 = df.fillna(colmean)

xcol = ['がく片長さ', 'がく片幅', '花弁長さ', '花弁幅']
x = df2[xcol]
t = df2['種類']

from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=2, random_state=0)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.3, random_state=0)

model.fit(x_train,y_train)
score = model.score(x_test, y_test)

# print(model.predict([[0.53, 0.58, 0.630000, 0.92]])) 'Iris-virginica'

# import pickle
# with open('irismodel.pkl', 'wb') as f:
#   pickle.dump(model, f)

# print(model.tree_.feature)
# print(model.tree_.threshold)

# print(model.tree_.value[1])
# print(model.tree_.value[3])
# print(model.tree_.value[4])
# print(model.classes_)
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
x_train.columns = ['gaku_nagasa', 'gaku_haba', 'kaben_nagasa', 'kaben_haba']
plot_tree(model, feature_names=x_train.columns, filled=True)
plt.show()