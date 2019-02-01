#/usr/bin/python3
# CRF model

from sklearn_crfsuite import CRF

from models.model_base import ModelBase, ModelFactory

class CRFfactory(ModelFactory):
    def get_model():
        return CRFmodel()

class CRFmodel(ModelBase):

    def __init__(self):
        self.model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=500,
            all_possible_transitions=True
        )

    def fit(self, X, Y):
        if self.debug:
            print("training CRF...")
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)
