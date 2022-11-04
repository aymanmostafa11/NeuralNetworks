from DataManager import prep_data
from models.NeuralNetworks import Perceptron


def train_button():
    prep_data()
    model = Perceptron()
    model.fit()
    # TODO: enable retrain and test buttons

def retrain_button():
    pass


def test_button():
    pass