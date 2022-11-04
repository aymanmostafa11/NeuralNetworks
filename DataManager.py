import pandas as pd
from sklearn.preprocessing import LabelEncoder
# TODO : add data reading, preprocessing

encoder = LabelEncoder()
filepath = "data/penguins.csv"


# TODO: create function to get train or test data

def prep_data():
    data = read()
    data = preprocessing(data, data.columns)
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


