#/usr/bin/python3
# NCRF model

from ncrfpp_mod.model.seqmodel import SeqModel
from ncrfpp_mod.utils.data import Data

from models.model_base import ModelBase, ModelFactory

class NCRFppFactory(ModelFactory):
    def get_model(params):
        return NCRFppModel(**params)

class NCRFppModel(ModelBase):

    should_encode = True

    def __init__(self, epochs=1, batch_size=1):
        self.data = Data()
        #self.model = SeqModel(self.data)
        self.epochs = int(epochs)

    def fit(self, X, Y):
        if self.debug:
            print("training NCRFpp...")
        self.data.build_alphabet(X, Y)
        self.data.fix_alphabet()
        print('ncrf char alphabet size:', self.data.char_alphabet_size)
        #self.data.generate_instance(X, Y, 'train')
        self.model = SeqModel(self.data)
        self.model.fit(X, Y, self.data, epochs=self.epochs)

    def predict(self, X):
        #self.data.generate_instance(X, Y=None, name='text')
        return self.model.predict(X, self.data)
