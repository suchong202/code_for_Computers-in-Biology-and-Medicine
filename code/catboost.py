import pandas as pd
import warnings
from sklearn.utils import shuffle
import numpy as np
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import catboost as cb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,accuracy_score
warnings.filterwarnings('ignore')

df0 = pd.read_excel('path',header =None)
df0['lable'] = 0
df1 = pd.read_excel('path',header =None)
df1['lable'] = 1
df2 = pd.read_excel('path',header =None)
df2['lable'] = 2
df3 = pd.read_excel('path',header =None)
df3['lable'] = 3

df = pd.concat([df0,df1,df2,df3])
colomes = df.columns[0:-1].values

df = shuffle(df)
df.index  = range(df.shape[0])
df.dropna(inplace=True)
X = df.loc[:,colomes]
Y = df['lable'].values.tolist()
X.astype('float64')
X = np.array(X)
#X = np.expand_dims(X,1)
Y = np.array(Y)
X_train,y_train = X,to_categorical(Y)
X_train.shape

df_0 = pd.read_excel('path',header =None)
df_0['lable'] = 0
df_1 = pd.read_excel('path',header =None)
df_1['lable'] = 1
df_2 = pd.read_excel('path',header =None)
df_2['lable'] = 2
df_3 = pd.read_excel('path',header =None)
df_3['lable'] = 3
df_t = pd.concat([df_0,df_1,df_2,df_3])

X_test = df_t.loc[:,colomes]
df_t.index  = range(df_t.shape[0])
y_test = df_t['lable'].values.tolist()
X_test.astype('float64')
X_test = np.array(X_test)



#X_train = np.array(X_train)

Y = np.array(Y)
X_train = np.array(X_train)


model = cb.CatBoostClassifier(eval_metric="AUC", depth=6, iterations=300,learning_rate=0.7)
skf = StratifiedKFold(n_splits=5,shuffle=False,random_state=0)

for train_index, test_index in skf.split(X_train,Y): 
    print("Train:", train_index, "Validation:", test_index) 
    X_tr, X_te = X_train[train_index], X_train[test_index] 
    Y_tr, Y_te = Y[train_index], Y[test_index]
    
   # model = LogisticRegression()
    model.fit(X_tr, Y_tr)
    result =model.predict(X_test)
    s = accuracy_score(y_test,result)
    print("score")
    print(s)
    
scores = cross_val_score(model,X_train,Y,cv=skf)

print("straitified cross validation scores:{}".format(scores))
print("Mean score of straitified cross validation:{:.5f}".format(scores.mean()))

score = accuracy_score(y_test, result)
print('Accuracy: %.5f' % score)

print(classification_report(y_test,result))