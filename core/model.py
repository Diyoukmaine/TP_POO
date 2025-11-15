from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
