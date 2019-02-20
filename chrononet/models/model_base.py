#/usr/bin/python3
# Model ModelBase

class ModelBase:
    debug = False
    model = None

class ModelFactory:
    requires_dim = False

    def get_model():
        return ModelBase()
