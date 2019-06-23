from abc import abstractmethod

class Model(object):
    @abstractmethod
    def act(self, observation, reward, done):
        raise NotImplementedError('All models must implement the "act" method')

    @abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError('All models must implement the "build" method')