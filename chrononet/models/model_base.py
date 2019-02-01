#/usr/bin/python3
# Model ModelBase

class ModelBase:
    debug = False
    model = None

class ModelFactory:
    def get_model():
        return ModelBase()
