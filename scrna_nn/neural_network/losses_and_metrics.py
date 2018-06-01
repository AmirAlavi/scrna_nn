from keras import backend as K
from theano import tensor as T

# LOSSES (used for parameter updates)
def get_dynamic_contrastive_loss(margin=1):
    def dynamic_contrastive_loss(y_true, y_pred):
        """y_true is a float between 0 and 1.0, instead of binary.
        (it acts as a way to dynamically (for each sample) modify the margin in 
        penalty, depending on how different the two samples in the pair actually are)
        """
        if y_true == 1:
            # Means that the distance in the ontology for this point was 0, exact same
            print("y_true == 1")
            return 0.5*K.square(y_pred)
        else:
            return 0.5*K.square(K.maximum((1-y_true)*margin - y_pred, 0))
    return dynamic_contrastive_loss

def get_triplet_batch_hard_loss(batch_size):
    def triplet_batch_hard_loss(y_true, y_pred):
        # y_pred is the embedding, y_true is the IDs (labels) of the samples (not 1-hot encoded)
        # They are mini-batched. If batch_size is B, and embedding dimension is D, shapes are:
        #   y_true: (B,)
        #   y_pred: (B,D)
        margin = 0.2
    
        # Get all-pairs distances
        y_true = K.sum(y_true, axis=1)
        diffs = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
        dist_mat = K.sqrt(K.sum(K.square(diffs), axis=-1) + K.epsilon())
        same_identity_mask = K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0))
        # TODO: make this backend-agnostic somehow
        negative_mask = T.bitwise_not(same_identity_mask)
        # XOR ensures that the same sample is paired with itself
        positive_mask = T.bitwise_xor(same_identity_mask, K.eye(batch_size, dtype='bool'))
        #print(K.int_shape(y_true))
        #print(K.int_shape(y_pred))

        #positive_mask = T.bitwise_xor(same_identity_mask, T.eye(K.int_shape(y_true)[0]))

        furthest_positive = K.max(dist_mat*positive_mask, axis=1)
        #closest_negative = K.min(dist_mat*negative_mask + np.inf*same_identity_mask, axis=1)
        closest_negative = K.min(dist_mat*negative_mask + 1e6*same_identity_mask, axis=1)

        loss = K.maximum(furthest_positive - closest_negative + margin, 0)
        return loss
    return triplet_batch_hard_loss

# METRICS (not used for parameter updates)
def get_frac_active_triplet_metric(batch_size):
    def frac_active_triplet_metric(y_true, y_pred):
        loss = get_triplet_batch_hard_loss(batch_size)(y_true, y_pred)
        num_active = K.sum(K.greater(loss, 1e-5))
        return num_active/batch_size
    return frac_active_triplet_metric
