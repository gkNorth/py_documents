import pandas as pd
from sklearn.preprocessing import StandardScaler #データ標準化
# csv読み込み
df = pd.read_csv('Boston.csv')
# 列ごとの平均値で欠損値の穴埋め
df2 = df.fillna(df.mean())
# ダミー変数化
dummy = pd.get_dummies(df2['CRIME'], drop_first=True) #CRIME列のダミー変数化
df3 = df2.join(dummy) #df2とdummyを列方向に結合
df3 = df3.drop(['CRIME'], axis=1) #元のCRIME列を削除
# データ標準化
df4 = df3.astype('float') #中身が整数だとfit_transformではエラー
sc = StandardScaler() #標準化関数代入
sc_df = sc.fit_transform(df4) #標準化

print(df3.head(3))