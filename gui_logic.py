from DataManager import prep_data
from models.NeuralNetworks import Perceptron


def train_button():
    data = prep_data()
    model = Perceptron(learning_rate=0.1, epochs=10)
    Y = data['body_mass_g']
    X = data.drop('body_mass_g', axis=1)
    model.fit(X, Y)

def retrain_button():
    pass


def test_button():
    pass