import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
filepath = "data/penguins.csv"


def prep_data(classes, features):
    """
    :return: x_train, x_test, y_train, y_test
    """
    data = read()
    featuresToDrop = [feature for feature in data.columns.drop("species") if feature not in features]
    data.drop(featuresToDrop, inplace=True, axis=1)
    data = preprocessing(data, data.columns.drop("species"))
    data = encodeTargets(data, classes)

    return split_and_shuffle(data)


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
    data.reset_index(drop=True, inplace=True)

    return data


def split_and_shuffle(data, equal_samples_per_class=False, train_size=0.7):
    """
    :return: x_train, x_test, y_train, y_test
    """
    data = data.sample(frac=1)  # suffle

    if equal_samples_per_class:
        return split(data)

    train_count = int(train_size * data.shape[0])

    train = data.iloc[:train_count, :]
    test = data.iloc[train_count:, :]

    x_train, y_train = train.drop(['species'], axis=1), train['species']
    x_test, y_test = test.drop(['species'], axis=1), test['species']

    return x_train, x_test, y_train, y_test


def split(data):
    """
    :return: x_train, x_test, y_train, y_test
    """
    trainData_for_class_0 = data[data["species"] == -1][:30]
    trainData_for_class_1 = data[data["species"] == 1][:30]
    trainData = trainData_for_class_0.append(trainData_for_class_1)

    X_train = trainData.drop("species", axis=1)
    Y_train = trainData["species"]

    testData_for_class_0 = data[data["species"] == -1][30:]
    testData_for_class_1 = data[data["species"] == 1][30:]
    testData = testData_for_class_0.append(testData_for_class_1)

    X_test = testData.drop("species", axis=1)
    Y_test = testData["species"]

    return X_train, X_test, Y_train, Y_test


def get_viz_data(classes, features):
    """
    :return: data without encoding targets and splitting
    """
    data = read()
    featuresToDrop = [feature for feature in data.columns.drop("species") if feature not in features]
    data.drop(featuresToDrop, inplace=True, axis=1)
    data = preprocessing(data, data.columns.drop("species"))
    data = data[(data['species'] == classes[0]) | (data['species'] == classes[1])]
    return data

