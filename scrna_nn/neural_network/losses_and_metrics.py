import numbers

from keras import backend as K
from theano import tensor as T
import numpy as np

from .. import util

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

def get_contrastive_batch_loss(batch_size, margin):
    try:
        margin = float(margin)
        print("Using hard-margin of {} in contrastive batch loss".format(margin))
        final_loss_tensor = lambda pos_dists, neg_dists: 0.5*K.square(pos_dists) + 0.5*K.square(K.maximum(margin-neg_dists, 0))
    except ValueError:
        raise util.ScrnaException('Contrastive margin must be a real number!')
    def contrastive_batch_loss(y_true, y_pred):
        # y_pred is the embedding, y_true is the IDs (labels) of the samples (not 1-hot encoded)
        # They are mini-batched. If batch_size is B, and embedding dimension is D, shapes are:
        #   y_true: (B,)
        #   y_pred: (B,D)
        
        # Get all-pairs distances
        y_true = K.sum(y_true, axis=1)
        diffs = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
        dist_mat = K.sqrt(K.sum(K.square(diffs), axis=-1) + K.epsilon())
        dist_mat = T.tril(dist_mat) # Keep only half the dist matrix (avoid double counting)
        same_identity_mask = K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0))
        negative_mask = T.bitwise_not(same_identity_mask)
        positive_mask = T.bitwise_xor(same_identity_mask, K.eye(batch_size, dtype='bool'))
        negative_mask = T.tril(negative_mask)
        positive_mask = T.tril(positive_mask)
        positive_distances = K.sum(dist_mat*positive_mask, axis=1) / (K.sum(positive_mask, axis=1) + K.epsilon())
        negative_distances = K.sum(dist_mat*negative_mask + 1e6*same_identity_mask, axis=1) / (K.sum(negative_mask, axis=1) + K.epsilon())

        loss = final_loss_tensor(positive_distances, negative_distances)
        return loss
    return contrastive_batch_loss

def get_triplet_batch_hard_loss(batch_size, margin):
    if margin == 'soft':
        print("Using soft-margin in batch-hard loss")
        final_loss_tensor = lambda hard_pos, hard_neg: K.softplus(hard_pos - hard_neg)
    else:
        try:
            margin = float(margin)
            print("Using hard-margin of {} in batch-hard loss".format(margin))
            final_loss_tensor = lambda hard_pos, hard_neg: K.maximum(hard_pos - hard_neg + margin, 0)
        except ValueError:
            raise util.ScrnaException('Batch hard margin must be a real number or "soft"!')
    def triplet_batch_hard_loss(y_true, y_pred):
        # y_pred is the embedding, y_true is the IDs (labels) of the samples (not 1-hot encoded)
        # They are mini-batched. If batch_size is B, and embedding dimension is D, shapes are:
        #   y_true: (B,)
        #   y_pred: (B,D)
    
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

        loss = final_loss_tensor(furthest_positive, closest_negative)
        return loss
    return triplet_batch_hard_loss

# METRICS (not used for parameter updates)
def get_frac_active_triplet_metric(batch_size, margin):
    def frac_active_triplet_metric(y_true, y_pred):
        loss = get_triplet_batch_hard_loss(batch_size, margin)(y_true, y_pred)
        num_active = K.sum(K.greater(loss, 1e-5))
        return num_active/batch_size
    return frac_active_triplet_metric

def embed_l2_metric(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred), axis=-1))

def get_embed_neg_dists_metric(batch_size):
    def embed_neg_dists_metric(y_true, y_pred):
        # Get all-pairs distances
        y_true = K.sum(y_true, axis=1)
        diffs = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
        dist_mat = K.sqrt(K.sum(K.square(diffs), axis=-1) + K.epsilon())
        same_identity_mask = K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0))
        negative_mask = T.bitwise_not(same_identity_mask)
        avg_negative_dists = K.sum(dist_mat*negative_mask, axis=-1) / K.maximum(K.sum(negative_mask, axis=1), 1)
        return avg_negative_dists
    return embed_neg_dists_metric

def get_embed_pos_dists_metric(batch_size):
    def embed_pos_dists_metric(y_true, y_pred):
        # Get all-pairs distances
        y_true = K.sum(y_true, axis=1)
        diffs = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
        dist_mat = K.sqrt(K.sum(K.square(diffs), axis=-1) + K.epsilon())
        same_identity_mask = K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0))
        positive_mask = T.bitwise_xor(same_identity_mask, K.eye(batch_size, dtype='bool'))
        avg_positive_dists = K.sum(dist_mat*positive_mask, axis=-1) / K.maximum(K.sum(positive_mask, axis=1), 1)
        return avg_positive_dists
    return embed_pos_dists_metric
