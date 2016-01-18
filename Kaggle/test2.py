# -*- coding: utf-8 -*-

import pandas as pd

def analysis(df):

   subdf = df[['Pclass','Sex','Fare', 'Embarked']]
   fare = subdf['Fare'].astype('float')
   fare[fare <= 8.75] = 0
   fare[(fare > 8.75) & (fare <= 32.2)] = 1
   fare[(fare > 32.2) & (fare <= 55.575)] = 2
   fare[fare > 55.575] = 3
   sub_fare = pd.get_dummies(fare, prefix='sub_Fare')
   pclass = pd.get_dummies(subdf['Pclass'],prefix='sub_Pclass')
   sex = (subdf['Sex']=='male').astype('int')
   embarked = subdf['Embarked'].astype('str')
   embarked[embarked == 'C'] = 0
   embarked[embarked == 'Q'] = 1
   embarked[embarked == 'S'] = 2
   embarked = embarked.astype('float')

   sub_em = pd.get_dummies(embarked, prefix='sub_embarked')
   X = pd.concat([pclass, sex, sub_fare, sub_em],axis=1)
   return X


df = pd.read_csv('train.csv')
X_train = analysis(df)
y_train = df.Survived


df_test = pd.read_csv('test.csv')
X_test = analysis(df_test)


#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=1)
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators = 100, criterion='entropy', min_samples_leaf=5)
clf = clf.fit(X_train,y_train)

result = clf.predict(X_test)

import csv as csv
predict_file = open("result.csv", "w", newline='')
predict_file_obj = csv.writer(predict_file)
predict_file_obj.writerow(["PassengerId", "Survived"])


for num in range(len(df_test['PassengerId'])):
   predict_file_obj.writerow([df_test['PassengerId'][num], result[num]])
predict_file.close()

#print("準確率為：{:.2f}".format(clf.score(X_test,y_test)))


'''
from sklearn import metrics
def measure_performance(X,y,clf, show_accuracy=True, 
                        show_classification_report=True, 
                        show_confusion_matrix=True):
    y_pred=clf.predict(X)
    print(y_pred)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")

    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y,y_pred),"\n")
        
    if show_confusion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y,y_pred),"\n")
        
measure_performance(X_test,y_test,clf, show_classification_report=True, show_confusion_matrix=True)
'''



import pydotplus
from io import StringIO
from sklearn import tree

dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data, feature_names=['Age', 'Sex','1st_class','2nd_class','3rd_class'])
tree.export_graphviz(clf, out_file=dot_data) 


pydotplus.graph_from_dot_data(dot_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('titanic2.png')

#from IPython.core.display import Image

#Image(filename='titanic.png')

