# 機械学習（データ分析）

## フロー

0. 目的設定・データ収集

1. データの前処理
  - ファイル読み込み
  - 欠損値の確認,補完（isnull, mean）
  - 外れ値の確認,削除（散布図）
  - 特徴量,正解データ取り出し
  
2. モデルの作成・学習
3. モデルの評価
4. 影響度の分析

## 対応方法

### ダミー変数化
列数が変わる為、データ分割前に実施しなければならない
  - df['xxx'].value_counts()
  - crime = pd.get_dummies(df['CRIME'], drop_first=True)
  - df2 = pd.concat([df, crime], axis=1)
  - df2 = df2.drop(['CRIME'], axis=1)
  
### データ分割（モデルテスト用）
訓練データ(training)、検証データ(validate)、テストデータ(test)の3つに分類する
※最初は[訓練・検証]データ と テストデータに分ける
  - train_val, test = train_test_split(df2, test_size=0.2, random_state=0)

### 欠損値
  - is_null()
  - any() : 列の有無のみ
  - sum() : 欠損値の個数
  - df.fillna(df.mean()) : 欠損値を平均値で穴埋め
  - df.median() : 中央値
  - df.dropna() : 欠損値がある行・列を削除
  - df.dropna(how='all', axis=0) : 全ての列で値が欠損値の行が削除
  - df.dropna(how='all', axis=1) : 全ての行で値が欠損値の列が削除
  - df.dropna(how='any) : 欠損値が1つでも含まれる行・列を削除（行or列はaxisで指定）
  - df['xxx']..value_counts() : 各値それぞれの出現回数
  
### 外れ値
  - グラフ kind='scatter'(散布図)参照
  - df2[ (df2['target'] > 100) & (df2['x2'] < -2 )] : 外れ値の参照
  - df3 = df2.drop(削除するインデックス, axis=0) : 削除
  - 削除対象は特徴量として利用する列のみにすると良い
  - 相関関係（右肩上がりor下がり）にあるものを特徴量とする（分散が大きい散布図は対象から外す）
  - 外れ値を全て削除すると、テストデータに外れ値が含まれる場合、予測性能が下がる
  
### データ操作（確認・結合）
  - df2.groupby('weather').mean()['cnt'] : weather列各値のcnt列平均値
  - df2[df2['dteday']=='2011-07-20'] : df2内でdtedayが2011-07-20である行
  - df2 = df.merge(weather, how='inner', on='weather_id')
  - how = inner, left, hist(alpha=0.5)
  - df3['atemp'] = df3['atemp'].interpolate() : 線形補間（オブジェクト型は補間できない）
  
  
### 相関関係
  - train_val4.corr() : 相関行列
  - train_val4.corr()['PRICE']
  - 正・負の相関関係があるのでabs()を使って絶対値に置き換える
  - シリーズ(配列).map(関数)でそれぞれに適用できる
  - abs_cor = train_cor.map(abs)
  - abs_cor.sort_values(ascending=False)  ascending?'昇順':'降順';
  - データフレーム型の場合 ↑ + 引数にby=列名
  
### データ分割（モデル学習用）
各データを特徴量, 正解データに分類する
  - x = df2.loc['行ラベル', '列ラベル']
  - df2.loc[ : , :'x3' ]
  - from sklearn.model_selection import train_test_split
  - x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)
  
### データ標準化
各特徴量の分布が大きく異なる　→　特徴量を1増加させるための労力が異なる
標準化を行うと、「平均値が0, 標準偏差が1」になる
  - from sklearn.preprocessing import StandardScaler
  - sc_model_x = StandardScaler()
  - sc_model_x.fit(x_train)
  - sc_x = sc_model_x.transform(x_train)
  - y_train(正解データ)も標準化する
  - 変数名.inverse_transform(標準化後のデータ) : 標準化前に逆変換
  - 検証データ or テストデータも標準化する
  - 上記それぞれで平均値や標準偏差を調べてはいけないので、sc_model_xを使う
  - sc_x_val = sc_model_x.transform(x_val)
  - sc_y_val = sc_model_y.transform(y_val)
  
### モデル
  - model.fit(x_train, y_train) : 訓練データ
  - model.score(x_test, y_test) : テストデータ
  - model.predict([xxx, xxxx]) : 未知データ
  - 重回帰モデル
    - from sklearn.linear_model import LinearRegression
    - model = LinearRegression()
  - 決定木モデル（Decision Tree）
    - from sklearn import tree
    - model = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
    - class_weight='balanced' : 正解データのバランスが悪い場合に使用する
  
### グラフ
  - import matplotlib.pyplot as plt
  - df.plot(kind='scatter', x='x0', y='target')
  - plt.show()
  - df['xxx'].plot(kind='bar')
  - kind : scatter, bar
  
### モデルの検証
  - チューニングしやすいようにモデル定義をlearn関数にする
  - 特徴量エンジニアリング
    - x['RM2'] = x['RM'] ** 2
    - df.loc[新しいインデックス名] = シリーズ