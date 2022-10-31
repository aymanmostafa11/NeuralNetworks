import pandas as pd
from sklearn.preprocessing import LabelEncoder
# TODO : add data reading, preprocessing

encoder = LabelEncoder()

def read(data):
    penguins = pd.read_csv(data)
    return penguins


def preprocessing(data,dataFeatures , test = False):
    for feature in dataFeatures:
        if data[feature].dtypes == object:
            data[feature].fillna(data[feature].mode()[0])
        else:
            data[feature].fillna(data[feature].mean())


   
    for feature in dataFeatures:
        if data[feature].dtypes == object:
            encoder.fit(data[feature])
            data[feature] = encoder.transform(data[feature])

   

                  

   

