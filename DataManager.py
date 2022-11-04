import pandas as pd
from sklearn.preprocessing import LabelEncoder
# TODO : add data reading, preprocessing

encoder = LabelEncoder()
filepath = "data/penguins.csv"


# TODO: create function to get train or test data

def prep_data():
    data = read()
    data = preprocessing(data, data.columns.drop("species"))
    return data


def read():
    penguins = pd.read_csv(filepath)
    return penguins


def preprocessing(data, dataFeatures, test=False):
    for feature in dataFeatures:
        if data[feature].dtypes == object:
            data[feature].fillna(data[feature].mode()[0])
        else:
            data[feature].fillna(data[feature].mean())
   
    for feature in dataFeatures:
        if data[feature].dtypes == object:
            encoder.fit(data[feature])
            data[feature] = encoder.transform(data[feature])

    return data


def encodeTargets(data,classes):
    notWantedClass = [c for c in set(data["species"]) if c not in classes]
    data["species"].replace(classes,[-1,1],inplace = True)
    data = data[data["species"]!= notWantedClass[0]]

    return data


def split(data,features):
       featuresToDrop = [feature for feature in  data.columns.drop("species") if feature not in features]
       data.drop(featuresToDrop,inplace = True , axis = 1)
       
       trainData_for_class_0 = data[data["species"] == -1][:30]
       trainData_for_class_1 = data[data["species"] == 1][:30] 
       trainData = trainData_for_class_0.append(trainData_for_class_1)
       
       X_train = trainData.drop("species",axis = 1)
       Y_train = trainData["species"]

       testData_for_class_0 = data[data["species"] == -1][30:]
       testData_for_class_1 = data[data["species"] == 1][30:] 
       testData = testData_for_class_0.append(testData_for_class_1)
       
       X_test = testData.drop("species",axis = 1)
       Y_test = testData["species"]

       return X_train,X_test,Y_train,Y_test
