# -*- coding: utf-8 -*-

import pandas as pd
df = pd.read_csv('train.csv')
subdf = df[['Pclass','Sex','Fare', 'Embarked']]
#y = df.Survived
y_train = df.Survived
pclass = pd.get_dummies(subdf['Pclass'],prefix='sub_Pclass')
sex = (subdf['Sex']=='male').astype('int')
#X = pd.concat([pclass,age,sex],axis=1)
X_train = pd.concat([pclass,age,sex],axis=1)


df_test = pd.read_csv('test.csv')
subdf_test = df_test[['Pclass', 'Sex', 'Fare', 'Embarked']]
t_pclass = pd.get_dummies(subdf_test['Pclass'],prefix='sub2_Pclass')
t_sex = (subdf_test['Sex']=='male').astype('int')
X_test = pd.concat([t_pclass,t_age,t_sex],axis=1)


from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5)
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


dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data, feature_names=['Age', 'Sex','1st_class','2nd_class','3rd_class'])
tree.export_graphviz(clf, out_file=dot_data) 

pydotplus.graph_from_dot_data(dot_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('titanic2.png')

#from IPython.core.display import Image 
#Image(filename='titanic.png')

