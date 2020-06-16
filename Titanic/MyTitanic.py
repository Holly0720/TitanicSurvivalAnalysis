# import the useful module
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

# show all columns
pd.set_option('display.max_columns',None)

# import the data
df_train = pd.read_csv("/Titanic/Titanictrain.csv")
df_test = pd.read_csv("/Titanic/Titanictest.csv")
# check the info
'''
df_train.head()
df_train.info()
df_train.describe()
df_train.columns.values
'''
# df_train_new['Sex'].value_counts() 

# drop unuseful columns
df_train_new=df_train.drop(['Name','Ticket','Cabin'],1)
# PassengerId as index
df_train_new.set_index('PassengerId',inplace=True)
# deal with NaN data
df_train_new['Embarked'].fillna('S',inplace=True)

# deal with the non-numerical data: sex, Embarked
df_train_new['Sex'] = df_train_new['Sex'].map({'male':0,'female':1}).astype('int')
df_train_new['Embarked'] = df_train_new['Embarked'].map({'C':0,'Q':1,'S':2}).astype('int')

# feature and label, maybe preprocessing firstly 
X = preprocessing.minmax_scale(df_train_new.drop(['Survived'],1))
X = np.nan_to_num(X,nan=-99)
y = np.array(df_train_new['Survived']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# learn the model by RandomForest
RFC = RandomForestClassifier(n_estimators=100, max_depth=5)
RFC.fit(X_train,y_train)
accuracy_RF = RFC.score(X_test,y_test)
print(accuracy_RF)

# cross_val_score
print(np.mean(cross_val_score(RFC,X,y,cv=5)))

# predict the test and output the result
df_test_new=df_test.drop(['Name','Ticket','Cabin'],1)
df_test_new.set_index('PassengerId',inplace=True)
df_test_new['Fare'].fillna(df_test_new['Fare'].median(),inplace=True)
        
df_test_new['Sex'] = df_test_new['Sex'].map({'male':0,'female':1}).astype('int')
df_test_new['Embarked'] = df_test_new['Embarked'].map({'C':0,'Q':1,'S':2}).astype('int')

test_data = preprocessing.minmax_scale(df_test_new)
test_data = np.nan_to_num(test_data,nan=-99)

df_test['Survived'] = RFC.predict(test_data.reshape(418,-1))
df_test[['PassengerId','Survived']].to_csv('/Titanic/submission.csv',index=False)
