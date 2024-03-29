import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()
mnist_encoder = OneHotEncoder()

filepath = "data/penguins.csv"
mnist_train_path = "data/mnist/mnist_train_mini.csv"
mnist_test_path = "data/mnist/mnist_test_mini.csv"
mnist_cache = None

def prep_data(dataset, classes, features, mlp_labels=False):
    """
    :param dataset:
    :return: x_train, x_test, y_train, y_test
    """

    if dataset == "penguins":
        return __prep_penguins(classes, features, mlp_labels)
    elif dataset == "mnist":
        return prep_mnist()
    else:
        raise ValueError(f"Data requested ({dataset}) not available")


def __prep_penguins(classes, features, mlp_labels):
    data = read()
    featuresToDrop = [feature for feature in data.columns.drop("species") if feature not in features]
    data.drop(featuresToDrop, inplace=True, axis=1)
    data = preprocessing(data, data.columns.drop("species"))

    data.iloc[:, 1:] = standardize(data.drop(['species'], axis=1))
    if mlp_labels:
        target = encode_multilabel_targets(data["species"])
        data.drop(["species"], axis=1, inplace=True)
        data["label_0"] = target[:, 0]
        data["label_1"] = target[:, 1]
        data["label_2"] = target[:, 2]
    else:
        data = encodeTargets(data, classes)

    return split_and_shuffle(data, mlp_labels)


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
            label_encoder.fit(data[feature])
            data[feature] = label_encoder.transform(data[feature])

    return data


def encodeTargets(data,classes):
    notWantedClass = [c for c in set(data["species"]) if c not in classes]
    data["species"].replace(classes, [-1, 1], inplace = True)
    data = data[data["species"] != notWantedClass[0]]
    data.reset_index(drop=True, inplace=True)

    return data


def encode_multilabel_targets(y):
    encoded_labels = one_hot_encoder.fit_transform(y.values.reshape(-1, 1))
    return encoded_labels.toarray()


def split_and_shuffle(data, mlp_labels=False, equal_samples_per_class=False, train_size=0.7):
    """
    :return: x_train, x_test, y_train, y_test
    """
    target_column = ["species"]
    if mlp_labels:
        target_column = ["label_0", "label_1", "label_2"]

    data = data.sample(frac=1)  # suffle

    if equal_samples_per_class:
        return split(data)

    train_count = int(train_size * data.shape[0])

    train = data.iloc[:train_count, :]
    test = data.iloc[train_count:, :]


    x_train, y_train = train.drop(target_column, axis=1), train[target_column]
    x_test, y_test = test.drop(target_column, axis=1), test[target_column]

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


def standardize(data):
    return (data - data.mean()) / data.std()


def get_viz_data(classes, features):
    """
    :return: data without encoding targets and splitting
    """
    data = read()
    featuresToDrop = [feature for feature in data.columns.drop("species") if feature not in features]
    data.drop(featuresToDrop, inplace=True, axis=1)
    data = preprocessing(data, data.columns.drop("species"))
    data = data[(data['species'] == classes[0]) | (data['species'] == classes[1])]
    data.iloc[:, 1:] = standardize(data.drop(['species'], axis=1))
    return data


def prep_mnist(reduceDimensions = True , degree = 80):
    global mnist_cache
    if mnist_cache is not None:
        return mnist_cache["x_train"], mnist_cache["x_test"], mnist_cache["y_train"], mnist_cache["y_test"]

    mnist_train, mnist_test = read_mnist()
    mnist_train.fillna(0,inplace = True)
    mnist_test.fillna(0,inplace = True)
    remove_constant_pixels(mnist_train)
    remove_constant_pixels(mnist_test)

    # Encode multilabel
    unique_labels = np.unique(mnist_train["label"])
    encoded_labels_train = mnist_encoder.fit_transform(mnist_train["label"].values.reshape(-1, 1)).toarray()
    mnist_train.drop(['label'], axis = 1, inplace=True)
    for i in range(len(unique_labels)):
        mnist_train["label_" + str(i)] = encoded_labels_train[:, i]

    encoded_labels_test = mnist_encoder.transform(mnist_test["label"].values.reshape(-1,1)).toarray()
    mnist_test.drop(["label"], axis = 1, inplace=True)
    for i in range(len(unique_labels)):
        mnist_test["label_" + str(i)] = encoded_labels_test[:, i]

    X_train , Y_train = split_mnist(mnist_train)
    X_test , Y_test = split_mnist(mnist_test)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    if reduceDimensions == True:
        X_train = reduce_dimensions_of_mnist(X_train, degree = degree)
        X_test = reduce_dimensions_of_mnist(X_test, degree = degree)


    mnist_cache = {"x_train": X_train, "x_test": X_test, "y_train": Y_train, "y_test": Y_test}

    return X_train, X_test, Y_train, Y_test


def read_mnist():
    mnist_train = pd.read_csv(mnist_train_path)
    mnist_test = pd.read_csv(mnist_test_path)
    return mnist_train, mnist_test


def remove_constant_pixels(data):
    for col in data:
        if data[col].max() == 0 or data[col].min() == 255:
            data.drop(columns=[col], inplace=True)    


def split_mnist(data):

    X = data.drop(columns=data.columns[-10:])
    Y = data.iloc[:, -10:]
    return X,Y


def reduce_dimensions_of_mnist(X,degree = 80):
    pca = PCA(n_components=degree)
    X_reduced = pca.fit_transform(X)
    return X_reduced


def get_encoder(type_ = "one_hot"):
    """

    :param type_: "one_hot" or "label"
    :return: encoder
    """
    if type_ == "one_hot":
        return one_hot_encoder
    elif type_ == "label":
        return label_encoder
    elif type_ == "mnist":
        return mnist_encoder
    else:
        raise ValueError("Encoder type doesn't exist, valid values are ('one_hot', 'label')")

