from sklearn.preprocessing import LabelEncoder
import pandas as pd


# One hot encoding using label encoder
def onehotencoding(data,catFeaturesList_,le,maxCatValues):

    cats = []
    for catFeature in catFeaturesList_:
        labels = data[catFeature].unique()
        if(len(labels) < maxCatValues):
            features = pd.get_dummies(data[catFeature])
            features.columns = catFeature + " = " + features.columns

            #print(catFeature)
            data = data.drop(catFeature,axis=1)
            data = pd.concat([data,features.astype(int)],axis=1)
        else:
            le.fit(labels)
            data[catFeature] = le.transform(data[catFeature])
    return(data)

def get_cat_features(data):
    catFeatureslist = []
    for colName,x in data.iloc[1,:].iteritems():
        if(str(x).isalpha()):
            catFeatureslist.append(colName)
    return(catFeatureslist)

def transform_cat_features(data,catFList_):
    for cf1 in catFList_:
        le = LabelEncoder()
        le.fit(data[cf1].unique())
        data[cf1] = le.transform(data[cf1])
    return(data)
