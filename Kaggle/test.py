from sklearn import tree
import pandas as pd
import numpy as np
from scipy.stats import mode

def cleandf(df):

    #cleaning fare column
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
    #classmeans = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
    #df.Fare = df[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )
    
    #cleaning the age column
    meanAge=np.mean(df.Age)
    df.Age=df.Age.fillna(meanAge)
    
    #cleaning the embarked column
    df.Cabin = df.Cabin.fillna('Unknown')
    #modeEmbarked = mode(df.Embarked)[0][0]
    #df.embarked = df.Embarked.fillna(modeEmbarked)

    return df

train_df = pd.read_csv("train.csv")

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X, Y)

train_df = cleandf(train_df)

#print(train_df.describe())

import pydot
from io import StringIO
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=['age','sex','1st_class','2nd_class','3rd_class']) 
dot_data.getvalue()
pydot.graph_from_dot_data(dot_data.getvalue())
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('titanic.png') 
from IPython.core.display import Image 
Image(filename='titanic.png')
