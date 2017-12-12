import math


def linear_decay(dist_in_ontology, lim=4):
    return max(0, 1 - (dist_in_ontology / lim))

def exponential_decay(dist_in_ontology, lim=4):
    if dist_in_ontology >= lim:
        return 0.0
    else:
        # Assumption: otherwise 0 <= distance < lim
        # Note: could use a decay constant (currently = 1)
        return math.exp(-1 * dist_in_ontology)
