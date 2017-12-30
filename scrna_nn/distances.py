import pickle
import math

from .util import ScrnaException


class PairSimilarity(object):
    def __init__(self, distance_mat_file, transform='linear', transform_param=None):
        with open(distance_mat_file, 'rb') as f:
            self.dist_mat = pickle.load(f)
        if transform not in ['linear', 'exponential', 'sigmoidal', 'binary']:
            raise ScrnaException("Bad similarity transform function!")
        if transform != 'linear' and transform !='binary' and transform_param == None:
            raise ScrnaException("Must provide value for transform_param!")
        self.transform = transform
        self.transform_param = transform_param

    def _binary_transform(self, a, b):
        if a == b:
            return 1
        else:
            return 0

    def _exponential_transform(self, x):
        return (math.pow(self.transform_param, x) - 1) / (self.transform_param - 1)

    def _sigmoidal_transform(self, x):
        print("hi!")
        num = 1 - math.exp(-self.transform_param * x)
        den = 1 - math.exp(-self.transform_param)
        return num / den

    def _transform(self, x, a, b):
        if self.transform == 'exponential':
            return self._exponential_transform(x)
        elif self.transform == 'sigmoidal':
            return self._sigmoidal_transform(x)
        elif self.transform == 'binary':
            return self._binary_transform(a, b)
        else:
            return x

    def __call__(self, a, b, transform=True):
        raise NotImplementedError

class OntologyBasedPairSimilarity(PairSimilarity):
    def __init__(self, max_ontology_distance, *args, **kwargs):
        self.max_dist = max_ontology_distance
        super().__init__(*args, **kwargs)

    def __call__(self, a, b, transform=True):
        dist = self.dist_mat[a][b]
        sim = max(0, 1 - (dist / self.max_dist))
        if transform:
            return self._transform(sim, a, b)
        return sim

class TextMinedPairSimilarity(PairSimilarity):
    def __call__(self, a, b, transform=True):
        if a == b: # necessary because the dictionary doesn't include distances to self
            return 1.0
        sim = self.dist_mat[a][b] # the value in dist_mat is already a similarity between 0 and 1
        if transform:
            return self._transform(sim, a, b)
        return sim

def linear_decay(dist_in_ontology, lim=4):
    return max(0, 1 - (dist_in_ontology / lim))

def exponential_decay(dist_in_ontology, lim=4):
    if dist_in_ontology >= lim:
        return 0.0
    else:
        # Assumption: otherwise 0 <= distance < lim
        # Note: could use a decay constant (currently = 1)
        return math.exp(-1 * dist_in_ontology)
