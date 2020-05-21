
from models.model_base import ModelBase, ModelFactory
from .pytorch_linker import DistanceClassifier

class TimeLinkerFactory(ModelFactory):
    def get_model(params):
        return TimeLinkerModel()

class TimeLinkerModel(ModelBase):

    def __init__(self):
        self.model = DistanceClassifier()

    def fit(self, X, Y):
        if self.debug:
            print('No training needed for distance classifier')

    def predict(self, X):
        anns, times = X
        return self.model.predict(anns, times)
