import pandas as pd
df = pd.read_csv('ex4.csv')
# sex_avg = df['sex'].mean()
# print(f'male:{round(sex_avg*100,1)}%  female:{100 - round(sex_avg*100,1)}%')

# print(df.groupby(['class', 'sex']).mean()['score'])

# print( pd.pivot_table(df, index='class', columns='sex', values='score', aggfunc=min) )

dept = pd.get_dummies(df['dept_id'], drop_first=True)
x = pd.concat([df, dept], axis=1)
df2 = x.drop('dept_id', axis=1)
print(df2)