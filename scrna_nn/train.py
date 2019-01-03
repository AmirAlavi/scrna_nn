# import pdb; pdb.set_trace()
import argparse
import datetime
import pickle
import time
from os.path import join

import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Lambda, Input
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from . import util
#from .data_manipulation.data_container import DataContainer, ExpressionSequence
from scrna_data_container import DataContainer, ExpressionSequence
from .neural_network import callbacks
from .neural_network import losses_and_metrics
from .neural_network import neural_nets as nn
from .neural_network import triplet
from .neural_network import unsupervised_pt as pt
from .neural_network.bio_knowledge import get_adj_mat_from_groupings
from .retrieval_test import retrieval_test_in_memory


def pretty_tdelta(tdelta):
    hours, rem = divmod(tdelta.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, mins, secs)


def get_model_architecture(
        working_dir_path,
        args,
        input_dim,
        output_dim):
    base_model = get_base_model_architecture(
        args, input_dim, output_dim)
    embedding_dim = base_model.layers[-1].input_shape[1]
    if args.nn == "DAE":
        embedding_dim = int(args.hidden_layer_sizes[-1])
    print(base_model.summary())
    # Set pretrained weights, if any, before making into siamese
    if args.init:
        nn.set_pretrained_weights(base_model, args.init)
    if args.freeze:
        nn.freeze_layers(base_model, args.freeze)
    if args.siamese:
        model = nn.get_siamese(base_model, input_dim, args.gn)
        # plot_model(model,
        #            to_file=join(working_dir_path, 'siamese_architecture.png'),
        #            show_shapes=True)
    elif args.triplet:
        model = nn.get_triplet(base_model)
    else:
        model = base_model
    return model, embedding_dim


def get_base_model_architecture(args, input_dim, output_dim):
    '''Possible options for neural network architectures are outlined in the
    '--help' command

    This function parses the user's options to determine what kind of
    architecture to construct. This could be a typical dense (MLP)
    architecture, a sparse architecture, or some combination. Users must
    provide an adjacency matrix for sparsely connected layers.
    '''
    adj_mat = None
    GO_adj_mats = None
    if args.nn == 'sparse':
        with open(args.sparse_groupings, 'rb') as f:
            adj_mat = pickle.load(f)
    if args.nn == 'GO':
        with open(args.go_arch, 'rb') as f:
            GO_adj_mats = pickle.load(f)

    hidden_layer_sizes = [int(x) for x in args.hidden_layer_sizes]

    return nn.get_nn_model(
        args,
        args.nn,
        hidden_layer_sizes,
        input_dim,
        args.act,
        output_dim,
        adj_mat,
        GO_adj_mats,
        args.with_dense,
        args.dropout)


def get_optimizer(args):
    if args.opt == 'sgd':
        print('Using SGD optimizer')
        lr = args.opt_lr
        decay = args.sgd_d
        momentum = args.sgd_m
        return SGD(
            lr=lr,
            decay=decay,
            momentum=momentum,
            nesterov=args.sgd_nesterov)
    elif args.opt == 'adam':
        print('Using Adam optimizer')
        return RMSprop(lr=args.opt_lr)
    elif args.opt == 'rmsprop':
        print('Using RMSprop optimizer')
        return Adam(lr=args.opt_lr)
    else:
        raise util.ScrnaException('Not a valid optimizer!')


def compile_model(model, args, optimizer):
    loss = None
    metrics = None
    if args.siamese:
        if args.dynMarginLoss:
            print('Using dynamic-margin contrastive loss')
            loss = losses_and_metrics.get_dynamic_contrastive_loss(
                args.dynMargin)
        else:
            print('Using contrastive loss')
            loss = nn.contrastive_loss
    elif args.triplet:
        batch_size = args.batch_hard_P * args.batch_hard_K
        margin = args.batch_hard_margin
        loss = losses_and_metrics.get_triplet_batch_hard_loss(batch_size, margin)
        frac_active_triplets = losses_and_metrics.get_frac_active_triplet_metric(
            batch_size, margin)
        avg_pos_dists = losses_and_metrics.get_embed_pos_dists_metric(batch_size)
        avg_neg_dists = losses_and_metrics.get_embed_neg_dists_metric(batch_size)
        metrics = [frac_active_triplets, avg_pos_dists, avg_neg_dists, losses_and_metrics.embed_l2_metric]
    elif args.nn == "DAE":
        loss = 'mean_squared_error'
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


def plot_accuracy_history(history, path):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(path)
    plt.close()


def train_pca_model(working_dir_path, args, data):
    print('Training a PCA model...')
    model = PCA(n_components=args.pca)
    X = data.get_expression_mat('train')
    model.fit(X)
    if not args.no_save:
        with open(join(working_dir_path, 'pca.p'), 'wb') as f:
            pickle.dump(model, f)
    return model


def fit_neural_net(model, args, data, callbacks_list, working_dir_path):
    if args.triplet:
        history = fit_triplet_neural_net(model, args, data, callbacks_list)
        triplet_net_metrics = ['frac_active_triplet_metric', 'embed_l2_metric', 'embed_pos_dists_metric', 'embed_neg_dists_metric']
        for metric in triplet_net_metrics:
            plt.figure()
            if metric == 'frac_active_triplet_metric':
                plt.semilogy(history.history[metric])
                plt.semilogy(history.history['val_'+metric])
            else:
                plt.plot(history.history[metric])
                plt.plot(history.history['val_'+metric])
            plt.title(metric)
            plt.xlabel('epoch')
            plt.legend(['train', 'valid'], loc='upper center')
            plt.savefig(join(working_dir_path, '{}.pdf'.format(metric)))
            plt.close()
    else:
        if args.siamese:
            # Specially routines for training siamese models
            data.create_siamese_data(args)
            X_train = data.splits['train']['siam_X']
            y_train = data.splits['train']['siam_y']
            X_valid = data.splits['valid']['siam_X']
            y_valid = data.splits['valid']['siam_y']
        else:
            if args.nn == "DAE":
                X_train, y_train = data.get_data_for_neural_net_unsupervised('train', args.noise_level)
                X_valid, y_valid = data.get_data_for_neural_net_unsupervised('valid', args.noise_level)
            else:
                X_train, y_train = data.get_data_for_neural_net(
                    'train', one_hot=True)
                X_valid, y_valid = data.get_data_for_neural_net(
                    'valid', one_hot=True)
            print('Train data shapes:')
            print(X_train.shape)
            print(y_train.shape)
            print('Valid data shapes:')
            print(X_valid.shape)
            print(y_valid.shape)
        train_sequence = ExpressionSequence(X_train, y_train, args.batch_size, "train")
        valid_sequence = ExpressionSequence(X_valid, y_valid, args.batch_size, "valid")
        del X_train, y_train, X_valid, y_valid
        history = model.fit_generator(train_sequence,
                                      steps_per_epoch=args.batches_per_epoch,
                                      epochs=args.epochs,
                                      callbacks=callbacks_list,
                                      validation_data=valid_sequence,
                                      verbose=2,
                                      shuffle=False)
        # history = model.fit(
        #     X_train,
        #     y_train,
        #     batch_size=args.batch_size,
        #     epochs=args.epochs,
        #     verbose=1,
        #     validation_data=(
        #         X_valid,
        #         y_valid),
        #     callbacks=callbacks_list)
    return history


def fit_triplet_neural_net(model, args, data, callbacks_list):
    print(model.summary())
    embedding_dim = model.layers[-1].output_shape[1]
    P = args.batch_hard_P
    K = args.batch_hard_K
    num_batches = args.num_batches
    num_batches_val = args.num_batches_val
    X_train, y_train = data.get_data_for_neural_net('train', one_hot=False)
    X_valid, y_valid = data.get_data_for_neural_net('valid', one_hot=False)
    train_data = triplet.TripletSequence(
        X_train, y_train, embedding_dim, P, K, num_batches)
    valid_data = triplet.TripletSequence(
        X_valid, y_valid, embedding_dim, P, K, num_batches_val)
    history = model.fit_generator(
        train_data,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=valid_data)
    return history


def save_neural_net(working_dir_path, args, model):
    print('saving model to folder: ' + working_dir_path)
    if args.checkpoints:
        path = join(working_dir_path, 'last_model.h5')
        path_weights = join(working_dir_path, 'last_model_weights.h5')
    else:
        path = join(working_dir_path, 'model.h5')
        path_weights = join(working_dir_path, 'model_weights.h5')
    if args.siamese:
        # For siamese nets, we only care about saving the subnetwork, not the
        # whole siamese net
        # For now, seems safe to assume index 2 corresponds to base net
        model = model.layers[2]
    print('Model saved:\n\n\n')
    print(model.summary())
    nn.save_trained_nn(model, path, path_weights)


def get_callbacks_list(working_dir_path, args):
    callbacks_list = []
    if args.sgd_step_decay:
        print('Using SGD Step Decay')
        lr_history = callbacks.StepLRHistory(args.sgd_lr, args.sgd_step_decay)
        lrate_sched = LearningRateScheduler(lr_history.get_step_decay_fcn())
        callbacks_list.extend([lr_history, lrate_sched])
    if args.early_stop_pat >= 0:
        callbacks_list.append(
            EarlyStopping(
                monitor=args.early_stop,
                patience=args.early_stop_pat,
                verbose=1,
                mode='min'))
    if args.early_stop_at_val >= 0:
        callbacks_list.append(
            callbacks.EarlyStoppingAtValue(
                monitor=args.early_stop_at,
                target=args.early_stop_at_val,
                verbose=1))
    if args.checkpoints:
        callbacks_list.append(
            ModelCheckpoint(
                working_dir_path +
                '/model.h5',
                monitor=args.checkpoints,
                verbose=1,
                save_best_only=True))
        callbacks_list.append(
            ModelCheckpoint(
                working_dir_path +
                '/model_weights.h5',
                monitor=args.checkpoints,
                verbose=1,
                save_best_only=True,
                save_weights_only=True))
    if args.loss_history:
        callbacks_list.append(callbacks.LossHistory(working_dir_path))
    return callbacks_list


def layerwise_train_neural_net(working_dir_path, args, data, training_report):
    print('Layerwise pre-training a Neural Network model...')
    # Set up optimizer
    opt = get_optimizer(args)
    # Get unlabeled data
    X = data.get_expression_mat()
    # Construct network architecture
    input_dim, output_dim = X.shape[1], None
    model, embed_dims = get_model_architecture(
        working_dir_path, args, input_dim, output_dim)
    training_report['cfg_DIMS'] = embed_dims
    # Greedy layerwise pretrain
    pt.pretrain_model(model, input_dim, opt, X, working_dir_path, args)

def train_LR(feature_model, args, data, training_report):
    lr = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='sag')
    for split in ['train', 'valid', 'test']:
        X, y = data.get_data_for_neural_net(split, one_hot=False)
        X, y_one_hot = data.get_data_for_neural_net(split, one_hot=True)
        X = feature_model.transform(X)
        if split == 'train':
            print("Fitting LR model")
            lr.fit(X, y)
        loss = log_loss(y_one_hot, lr.predict_proba(X))
        acc = accuracy_score(y, lr.predict(X))
        training_report['res_{}_loss'.format(split)] = loss
        print('{}\tLR loss\t{}'.format(split, loss))
        training_report['res_{}_acc'.format(split)] = acc
        print('{}\tLR acc\t{}'.format(split, acc))
    
def evaluate_pca_model(model, args, data, training_report):
    # Use the principal components as input features for Logistic Regression clf
    train_LR(model, args, data, training_report)
    # Retrieval testing
    print("Conducting retrieval testing...")
    database = data.get_expression_mat(split='train')
    database = model.transform(database)
    database_labels = data.get_labels('train')
    for split in ['valid', 'test']:
        query = data.get_expression_mat(split)
        query = model.transform(query)
        query_labels = data.get_labels(split)
        avg_map, wt_avg_map, avg_mafp, wt_avg_mafp = retrieval_test_in_memory(
            database, database_labels, query, query_labels)
        training_report['res_{}_avg_map'.format(split)] = avg_map
        training_report['res_{}_wt_avg_map'.format(split)] = wt_avg_map
        training_report['res_{}_avg_mafp'.format(split)] = avg_mafp
        training_report['res_{}_wt_avg_mafp'.format(split)] = wt_avg_mafp
        print("{}\tAvg MAP\t{}".format(split, avg_map))
        print("{}\tWt Avg MAP\t{}".format(split, wt_avg_map))
        print("{}\tAvg MAFP\t{}".format(split, avg_mafp))
        print("{}\tWt Avg MAFP\t{}".format(split, wt_avg_mafp))
    
    
def evaluate_model(model, args, data, training_report):
    # Get performance on each metric for each split
    if args.checkpoints:
        # if checkpointing was used, then make sure we use the 'best'
        # model for evaluation
        model.load_weights(join(training_report['cfg_folder'], 'model_weights.h5'))
    for split in data.splits.keys():
        if args.siamese:
            X = data.splits[split]['siam_X']
            y = data.splits[split]['siam_y']
        else:
            if args.nn == "DAE":
                X, y = data.get_data_for_neural_net_unsupervised(split, args.noise_level)
            else:
                X, y = data.get_data_for_neural_net(split, one_hot=not args.triplet)
        if args.triplet:
            # TODO: remove copy/pasted code
            embedding_dim = model.layers[-1].output_shape[1]
            bh_P = args.batch_hard_P
            bh_K = args.batch_hard_K
            num_batches = args.num_batches_val
            eval_data = triplet.TripletSequence(
                X, y, embedding_dim, bh_P, bh_K, num_batches)
            eval_results = model.evaluate_generator(eval_data, verbose=1)
        else:
            eval_results = model.evaluate(x=X, y=y)
        try:  # Ensure eval_results is iterable
            _ = (i for i in eval_results)
        except TypeError:
            eval_results = [eval_results]
        for metric, res in zip(model.metrics_names, eval_results):
            training_report['res_{}_{}'.format(split, metric)] = res
            print('{}\t{}\t{}'.format(split, metric, res))
    # Additionally, test retrieval performance with valid and test splits as
    # queries
    print("Conducting retrieval testing...")
    database = data.get_expression_mat(split='train')
    if args.nn == "DAE":
        sample_in = Input(shape=model.layers[0].input_shape[1:],
                          name='sample_input')
        embedded = Lambda(lambda x: model.layers[1].encode(x),
                          output_shape=(training_report['cfg_DIMS'],),
                          name='encoder')(sample_in)
        embedded._uses_learning_phase = True
        embedder = Model(sample_in, embedded)
    else:
        reducing_model = model
        if args.siamese:
            reducing_model = model.layers[2]
            last_hidden_layer = reducing_model.layers[-1]
        elif args.triplet:
            last_hidden_layer = reducing_model.layers[-1]
        else:
            last_hidden_layer = reducing_model.layers[-2]
        embedder = Model(inputs=reducing_model.layers[0].input, outputs=last_hidden_layer.output)
    database = embedder.predict(database)
    database_labels = data.get_labels('train')
    for split in ['valid', 'test']:
        query = data.get_expression_mat(split)
        query = embedder.predict(query)
        query_labels = data.get_labels(split)
        avg_map, wt_avg_map, avg_mafp, wt_avg_mafp = retrieval_test_in_memory(
            database, database_labels, query, query_labels)
        training_report['res_{}_avg_map'.format(split)] = avg_map
        training_report['res_{}_wt_avg_map'.format(split)] = wt_avg_map
        training_report['res_{}_avg_mafp'.format(split)] = avg_mafp
        training_report['res_{}_wt_avg_mafp'.format(split)] = wt_avg_mafp
        print("{}\tAvg MAP\t{}".format(split, avg_map))
        print("{}\tWt Avg MAP\t{}".format(split, wt_avg_map))
        print("{}\tAvg MAFP\t{}".format(split, avg_mafp))
        print("{}\tWt Avg MAFP\t{}".format(split, wt_avg_mafp))

def train_neural_net(working_dir_path, args, data, training_report):
    print('Training a Neural Network model...')
    # Construct network architecture
    ngpus = args.ngpus
    input_dim, output_dim = data.get_in_out_dims()
    if ngpus > 1:
        import tensorflow as tf
        with tf.device('/cpu:0'):
            template_model, embed_dims = get_model_architecture(
                working_dir_path, args, input_dim, output_dim)
        model = multi_gpu_model(template_model, gpus=ngpus)
    else:
        template_model, embed_dims = get_model_architecture(
            working_dir_path, args, input_dim, output_dim)
        model = template_model
    training_report['cfg_DIMS'] = embed_dims
    # Set up optimizer
    opt = get_optimizer(args)
    # Compile the model
    compile_model(model, args, opt)
    print(model.summary())
    print('model compiled and ready for training')
    # Prep callbacks
    callbacks_list = get_callbacks_list(working_dir_path, args)
    # Maybe add Plotter callback
    if args.triplet and args.plotter is not None:
        print("Adding a Plotter callback")
        if args.pca_plotter is not None:
            callbacks_list.append(
                callbacks.PCAPlotter(
                    args.pca_plotter,
                    model,
                    data=args.plotter,
                    out_dir=join(working_dir_path, 'pca_plotter'),
                    interval=args.plotter_int,
                    sample_normalize=args.sn,
                    feature_normalize=args.gn,
                    feature_mean=data.mean,
                    feature_std=data.std,
                    minmax_normalize=args.mn,
                    minmax_scaler=data.minmax_scaler
                )
            )
        else:
            callbacks_list.append(
                callbacks.TSNEPlotter(
                    model,
                    data=args.plotter,
                    out_dir=join(working_dir_path, 'tsne_plotter'),
                    interval=args.plotter_int,
                    sample_normalize=args.sn,
                    feature_normalize=args.gn,
                    feature_mean=data.mean,
                    feature_std=data.std,
                    minmax_normalize=args.mn,
                    minmax_scaler=data.minmax_scaler
                )
            )
    # Fit the model
    print('training model...')
    t0 = datetime.datetime.now()
    history = fit_neural_net(
        model,
        args,
        data,
        callbacks_list,
        working_dir_path)
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1 - t0)
    print('Training neural net took ' + time_str)
    training_report['res_train_time'] = time_str
    # Evaluate model
    # TODO: make this automatically happen via callback
    if not args.siamese and not args.triplet and args.nn != "DAE": # TODO: just do this by checking if 'acc' is a current metric
        plot_accuracy_history(history, join(working_dir_path, 'accuracy.png'))
    print('Evaluating')
    # Since evaluation functions may need to change the model,
    # save the model here first
    if not args.no_save:
        save_neural_net(working_dir_path, args, template_model)
    # Also save the mapping of label to string:
    with open(join(working_dir_path, "label_to_int_map.pickle"), 'wb') as f:
        pickle.dump(data.label_to_int_map, f)
    if not args.no_eval:
        evaluate_model(model, args, data, training_report)
    


def report_config(args, training_report):
    # Data normalization
    if args.sn:
        training_report['cfg_normalization'] = 'sn'
    elif args.gn:
        training_report['cfg_normalization'] = 'gn'
    elif args.mn:
        training_report['cfg_normalization'] = 'mn'
        training_report['cfg_normalization_mn_min'] = args.minmax_min
        training_report['cfg_normalization_mn_max'] = args.minmax_max
    else:
        training_report['cfg_normalization'] = 'none'
    # Rest of configuration space not relevant to PCA
    if training_report['cfg_type'] == 'pca':
        return

    training_report['cfg_epochs'] = args.epochs
    training_report['cfg_batch_size'] = args.batch_size
    training_report['cfg_activation'] = args.act
    if args.dropout > 0:
        training_report['cfg_dropout'] = args.dropout
    if args.l1_reg > 0:
        training_report['cfg_l1_reg'] = args.l1_reg
    if args.l2_reg > 0:
        training_report['cfg_l2_reg'] = args.l2_reg
    if args.nn == "DAE":
        training_report['cfg_noise_level'] = args.noise_level
    if args.with_dense > 0:
        training_report['cfg_with_dense'] = args.with_dense
    if args.freeze:
        training_report['cfg_freeze_n'] = args.freeze
    training_report['cfg_init'] = 'random'
    if args.init:
        training_report['cfg_init'] = args.init
    # Optimizer
    training_report['cfg_opt'] = args.opt
    training_report['cfg_lr'] = args.opt_lr
    if args.opt == 'sgd':
        training_report['cfg_decay'] = args.sgd_d
        training_report['cfg_momentum'] = args.sgd_m
        training_report['cfg_nesterov?'] = 'Y' if args.sgd_nesterov else 'N'
    if args.early_stop_pat >= 0:
        training_report['cfg_early_stop_patience'] = args.early_stop_pat
        training_report['cfg_early_stop_metric'] = args.early_stop
    if args.checkpoints:
        training_report['cfg_checkpoints'] = args.checkpoints
    # Siamese
    if args.siamese:
        training_report['cfg_siam?'] = 'Y'
        training_report['cfg_siam_unif_diff'] = args.unif_diff
        training_report['cfg_siam_same_lim'] = args.same_lim
        training_report['cfg_siam_diff_multiplier'] = args.diff_multiplier
        if args.dynMarginLoss:
            training_report['cfg_siam_pair_distances'] = args.dynMarginLoss
            training_report['cfg_siam_contrastive_margin'] = args.dynMargin
            training_report['cfg_siam_distances_source'] = args.dist_mat_file
            training_report['cfg_siam_distance_transform'] = args.trnsfm_fcn
            training_report['cfg_siam_distance_transform_param'] = args.trnsfm_fcn_param
    # Triplet
    if args.triplet:
        training_report['cfg_triplet?'] = 'Y'
        training_report['cfg_triplet_P'] = args.batch_hard_P
        training_report['cfg_triplet_K'] = args.batch_hard_K
        training_report['cfg_triplet_batches'] = args.num_batches


def load_data(args, working_dir):
    if args.layerwise_pt:
        data = DataContainer(
            join(
                args.data,
                'unlabeled_data.h5'),
            sample_normalize=args.sn,
            feature_normalize=args.gn,
            minmax_normalize=args.mn,
            min=args.minmax_min,
            max=args.minmax_max)
    else:
        data = DataContainer(
            join(
                args.data,
                'train_data'),
            sample_normalize=args.sn,
            feature_normalize=args.gn,
            minmax_normalize=args.mn,
            min=args.minmax_min,
            max=args.minmax_max)
        data.add_split(join(args.data, 'valid_data'), 'valid')
        data.add_split(join(args.data, 'test_data'), 'test')
    if args.gn:
        # save the training data mean and std for later use on new data
        data.mean.to_pickle(join(working_dir, 'mean.p'))
        data.std.to_pickle(join(working_dir, 'std.p'))
    if args.mn:
        with open(join(working_dir, 'minmax_scaler.p'), 'wb') as f:
            pickle.dump(data.minmax_scaler, f)
    return data


def train(args: argparse.Namespace):
    model_type = args.nn if args.nn is not None else 'pca'
    # create a unique working directory for this model
    working_dir_path = util.create_working_directory(
        args.out, 'models/', model_type)
    # with open(join(working_dir_path, 'command_line_args.json'), 'w') as fp:
    #     json.dump(args, fp)
    util.cli.save_cmd_args_to_file(
        join(working_dir_path, 'command_line_args.txt'))

    training_report = {'cfg_type': model_type, 'cfg_folder': working_dir_path}
    report_config(args, training_report)
    print('loading data and setting up model...')
    data = load_data(args, working_dir_path)
    if args.pca:
        model = train_pca_model(working_dir_path, args, data)
        if not args.no_eval:
            evaluate_pca_model(model, args, data, training_report)
        training_report['cfg_DIMS'] = args.pca
    elif args.nn:
        if args.layerwise_pt:
            layerwise_train_neural_net(
                working_dir_path, args, data, training_report)
        else:
            train_neural_net(working_dir_path, args, data, training_report)

    # Report the configuration and performance of the model
    with open(join(working_dir_path, 'config_results.csv'), 'w') as f:
        for i, col in enumerate(sorted(training_report.keys())):
            if i == len(training_report) - 1:
                f.write('{}\n'.format(col))
            else:
                f.write('{},'.format(col))
        for i, key in enumerate(sorted(training_report.keys())):
            if i == len(training_report) - 1:
                f.write('{}\n'.format(training_report[key]))
            else:
                f.write('{},'.format(training_report[key]))
