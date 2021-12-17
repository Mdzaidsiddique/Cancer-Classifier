import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

cancer = pd.read_csv('C:/Users/masoo/Downloads/breast-cancer-wisconsin.data', header=None, na_values='?')
cancer.head()
cancer.columns
cancer.columns   =["sample_code","thick","uni_cell_size","uni_cell_shape","mar_adh","Epi_cell_size","Bare_nuclei","bland_chro","normal_nuc","mitoses","Cancer_class"]  
cancer.shape
cancer.isnull().sum()
cancer.Bare_nuclei.fillna(cancer.Bare_nuclei.mean(),inplace = True)
cancer.info()
plt.boxplot(cancer)
cancer.sample_code.describe()
cancer.Cancer_class.unique()
cancer.Cancer_class= cancer.Cancer_class.replace(2, 'benign').replace(4, 'malignant')

x = cancer.iloc[:,:-1]
x.shape
x.columns
y = cancer.Cancer_class
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.20, random_state = 123)

#KNeighborsClassifier
knmodel = KNeighborsClassifier(18)
knmodel.fit(X = x_train, y = y_train)
knmodel.score(x_test,y_test) #0.5785714285714286
pred = knmodel.predict(x_test)
#confusion metrics
confusion_matrix(y_test, pred)

# max k value = no of obs of train data
# k = sqrt(n/2) best practice
from math import sqrt
print(sqrt(699/2))

#for loop for best k value
result = pd.DataFrame(columns = [ "k", "score_test", "score_train"])
for k in range(1, 619):
    knmodel = KNeighborsClassifier(k)
    knmodel.fit(x_train,y_train)
    knmodel.score(x_test,y_test)
    result = result.append({ "k" : k, "score_test" : knmodel.score(x_test,y_test) , "score_train" :knmodel.score(x_train,y_train)  },ignore_index=True)
plt.plot(result.k,result.score_test)
plt.plot(result.k,result.score_train)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
Dtmodel = DecisionTreeClassifier()
Dtmodel.fit(X = x_train, y = y_train)
Dtmodel.score(x_test, y_test) #0.9428571428571428
Dtmodel.score(x_train, y_train)
pred = Dtmodel.predict(x_test)
confusion_matrix(y_test, pred)

#displaying the decision tree
from sklearn import tree
tree.plot_tree(Dtmodel)


#ensemble learning - Together of many trees
from sklearn.ensemble import RandomForestClassifier
#random forest
rfmodel = RandomForestClassifier()
rfmodel.fit(X = x_train, y = y_train)
rfmodel.score(x_test, y_test) #0.9785714285714285
pred = rfmodel.predict(x_test)
confusion_matrix(y_test, pred)

from sklearn.ensemble import GradientBoostingClassifier as GB
gbmodel = GB()
gbmodel.fit(x_train,y_train)
gbmodel.score(x_test,y_test) #testing accuracy
gbmodel.score(x_train,y_train) #training accuracy

#xtreme Gredient Boosting 
from xgboost import XGBClassifier
xgmodel = XGBClassifier()
xgmodel.fit(X_train,y_train)
xgmodel.score(X_test,y_test) #testing accuracy

#adaboost - adaptive boosting

from sklearn.ensemble import AdaBoostClassifier
abmodel = AdaBoostClassifier()
abmodel.fit(x_train,y_train)
abmodel.score(x_test,y_test) #testing accuracy
abmodel.score(x_train,y_train) #training accuracy


#random forest isbest model
#final model
final_model = RandomForestClassifier()
final_model.fit(x,y)
final_model.score(x,y)

to_pred = {''}
















