class ModelNotFoundError(Exception):
    def __init__(self, path):
        super().__init__("Model {} was not found".format(path))
