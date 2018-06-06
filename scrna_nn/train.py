# import pdb; pdb.set_trace()
import argparse
import datetime
import pickle
import time
from os.path import join

import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from sklearn.decomposition import PCA

from . import util
from .data_manipulation.data_container import DataContainer
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
        output_dim,
        gene_names):
    base_model = get_base_model_architecture(
        args, input_dim, output_dim, gene_names)
    embedding_dim = base_model.layers[-1].input_shape[1]
    if args.nn == "DAE":
        embedding_dim = int(args.hidden_layer_sizes[0])
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


def get_base_model_architecture(args, input_dim, output_dim, gene_names):
    '''Possible options for neural network architectures are outlined in the
    '--help' command

    This function parses the user's options to determine what kind of
    architecture to construct. This could be a typical dense (MLP)
    architecture, a sparse architecture, or some combination. Users must
    provide an adjacency matrix for sparsely connected layers.
    '''
    adj_mat = None
    go_first_level_adj_mat = None
    go_other_levels_adj_mats = None
    flatGO_ppitf_adj_mats = None
    if args.nn == 'sparse' or args.nn == 'GO_ppitf':
        _, _, adj_mat = get_adj_mat_from_groupings(
            args.sparse_groupings, gene_names)
        print('Sparse layer adjacency mat shape: ', adj_mat.shape)
    if args.nn == 'GO' or args.nn == 'GO_ppitf':
        # For now, we expect these file names
        # TODO: decouple file naming
        go_first_level_groupings_file = join(
            args.go_arch, 'GO_arch_first_level_groupings.txt')
        t0 = time.time()
        _, _, go_first_level_adj_mat = get_adj_mat_from_groupings(
            go_first_level_groupings_file, gene_names)
        print('get adj mat from groupings file took: ', time.time() - t0)
        print(
            '(GO first level) Sparse layer adjacency mat shape: ',
            go_first_level_adj_mat.shape)
        go_other_levels_adj_mats_file = join(
            args.go_arch, 'GO_arch_other_levels_adj_mats.pickle')
        with open(go_other_levels_adj_mats_file, 'rb') as fp:
            go_other_levels_adj_mats = pickle.load(fp)
    elif args.nn == 'flatGO_ppitf':
        _, _, flatGO_adj_mat = get_adj_mat_from_groupings(
            args.fGO_ppitf_grps.split(',')[0], gene_names)
        _, _, ppitf_adj_mat = get_adj_mat_from_groupings(
            args.fGO_ppitf_grps.split(',')[1], gene_names)
        flatGO_ppitf_adj_mats = [flatGO_adj_mat, ppitf_adj_mat]
    # elif args['--nn'] == 'GO_ppitf':
    #     _, _, adj_mat = get_adj_mat_from_groupings(args['--sparse_groupings'], gene_names)

    hidden_layer_sizes = [int(x) for x in args.hidden_layer_sizes]
    # if args['--ae']:
    #     if args['--nn'] == 'GO':
    #         print('For GO autoencoder, doing 1st layer')
    #         adj_mat = go_first_level_adj_mat

    return nn.get_nn_model(
        args,
        args.nn,
        hidden_layer_sizes,
        input_dim,
        args.act,
        output_dim,
        adj_mat,
        go_first_level_adj_mat,
        go_other_levels_adj_mats,
        flatGO_ppitf_adj_mats,
        args.with_dense,
        args.dropout)


def get_optimizer(args):
    if args.opt == 'sgd':
        print('Using SGD optimizer')
        lr = args.sgd_lr
        decay = args.sgd_d
        momentum = args.sgd_m
        return SGD(
            lr=lr,
            decay=decay,
            momentum=momentum,
            nesterov=args.sgd_nesterov)
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
        loss = losses_and_metrics.get_triplet_batch_hard_loss(batch_size)
        frac_active_triplets = losses_and_metrics.get_frac_active_triplet_metric(
            batch_size)
        metrics = [frac_active_triplets]
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
    with open(join(working_dir_path, 'pca.p'), 'wb') as f:
        pickle.dump(model, f)


def fit_neural_net(model, args, data, callbacks_list, working_dir_path):
    if args.triplet:
        history = fit_triplet_neural_net(model, args, data, callbacks_list)
        plt.figure()
        plt.semilogy(history.history['frac_active_triplet_metric'])
        plt.title('Fraction of active triplets per epoch')
        plt.ylabel('% active triplets')
        plt.xlabel('epoch')
        plt.savefig(join(working_dir_path, 'frac_active_triplets.png'))
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
        history = model.fit(
            X_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            validation_data=(
                X_valid,
                y_valid),
            callbacks=callbacks_list)
    return history


def fit_triplet_neural_net(model, args, data, callbacks_list):
    print(model.summary())
    embedding_dim = model.layers[-1].output_shape[1]
    P = args.batch_hard_P
    K = args.batch_hard_K
    num_batches = args.num_batches
    X_train, y_train = data.get_data_for_neural_net('train', one_hot=False)
    X_valid, y_valid = data.get_data_for_neural_net('valid', one_hot=False)
    train_data = triplet.TripletSequence(
        X_train, y_train, embedding_dim, P, K, num_batches)
    valid_data = triplet.TripletSequence(
        X_valid, y_valid, embedding_dim, P, K, num_batches)
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
    # Construct network architecture
    gene_names = data.get_gene_names()
    input_dim, output_dim = data.get_in_out_dims()
    model, embed_dims = get_model_architecture(
        working_dir_path, args, input_dim, output_dim, gene_names)
    training_report['cfg_DIMS'] = embed_dims
    # Get unlabeled data
    X = data.get_expression_mat()
    # Greedy layerwise pretrain
    hidden_layer_sizes = [int(x) for x in args.hidden_layer_sizes]
    if args.nn == 'dense':
        if hidden_layer_sizes == [1136, 100]:
            pt.pretrain_dense_1136_100_model(
                model, input_dim, opt, X, working_dir_path, args)
        elif hidden_layer_sizes == [1136, 500, 100]:
            pt.pretrain_dense_1136_500_100_model(
                input_dim, opt, X, working_dir_path, args)
        else:
            raise util.ScrnaException(
                'Layerwise pretraining not implemented for this architecture')
    elif args.nn == 'sparse' and args.with_dense == 100:
        if 'flat' in args.sparse_groupings:
            print('Layerwise pretraining for FlatGO')
            if hidden_layer_sizes == [100]:
                _, _, adj_mat = get_adj_mat_from_groupings(
                    args.sparse_groupings, gene_names)
                pt.pretrain_flatGO_400_100_model(
                    input_dim, adj_mat, opt, X, working_dir_path, args)
            elif hidden_layer_sizes == [200, 100]:
                _, _, adj_mat = get_adj_mat_from_groupings(
                    args.sparse_groupings, gene_names)
                pt.pretrain_flatGO_400_200_100_model(
                    input_dim, adj_mat, opt, X, working_dir_path, args)
            else:
                raise util.ScrnaException(
                    'Layerwise pretraining not ' +
                    'implemented for this architecture')
        else:
            print('Layerwise pretraining for PPITF')
            if hidden_layer_sizes == [100]:
                _, _, adj_mat = get_adj_mat_from_groupings(
                    args.sparse_groupings, gene_names)
                pt.pretrain_ppitf_1136_100_model(
                    input_dim, adj_mat, opt, X, working_dir_path, args)
            elif hidden_layer_sizes == [500, 100]:
                _, _, adj_mat = get_adj_mat_from_groupings(
                    args.sparse_groupings, gene_names)
                pt.pretrain_ppitf_1136_500_100_model(
                    input_dim, adj_mat, opt, X, working_dir_path, args)
            else:
                raise util.ScrnaException(
                    'Layerwise pretraining not ' +
                    'implemented for this architecture')

    elif args.nn == 'GO' and args.with_dense == 31:
        go_first_level_groupings_file = join(
            args.go_arch, 'GO_arch_first_level_groupings.txt')
        _, _, go_first_level_adj_mat = get_adj_mat_from_groupings(
            go_first_level_groupings_file, gene_names)
        go_other_levels_adj_mats_file = join(
            args.go_arch, 'GO_arch_other_levels_adj_mats.pickle')
        with open(go_other_levels_adj_mats_file, 'rb') as fp:
            go_other_levels_adj_mats = pickle.load(fp)
        pt.pretrain_GOlvls_model(
            input_dim,
            go_first_level_adj_mat,
            go_other_levels_adj_mats[0],
            go_other_levels_adj_mats[1],
            opt,
            X,
            working_dir_path,
            args)
    else:
        raise util.ScrnaException(
            'Layerwise pretraining not implemented for this architecture')


def evaluate_model(model, args, data, training_report):
    # Get performance on each metric for each split
    for split in data.splits.keys():
        if args.siamese:
            X = data.splits[split]['siam_X']
            y = data.splits[split]['siam_y']
        else:
            X, y = data.get_data_for_neural_net(
                split, one_hot=not args.triplet)
        if args.triplet:
            # TODO: remove copy/pasted code
            embedding_dim = model.layers[-1].output_shape[1]
            bh_P = args.batch_hard_P
            bh_K = args.batch_hard_K
            num_batches = args.num_batches
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
    database = data.get_expression_mat(split='train')
    reducing_model = model
    if args.siamese:
        reducing_model = model.layers[2]
        last_hidden_layer = reducing_model.layers[-1]
    elif args.triplet:
        last_hidden_layer = reducing_model.layers[-1]
    else:
        last_hidden_layer = reducing_model.layers[-2]
    get_activations = K.function([reducing_model.layers[0].input], [
                                 last_hidden_layer.output])
    database = get_activations([database])[0]
    database_labels = data.get_labels('train')
    for split in ['valid', 'test']:
        query = data.get_expression_mat(split)
        query = get_activations([query])[0]
        query_labels = data.get_labels(split)
        avg_map, wt_avg_map, avg_mafp, wt_avg_mafp = retrieval_test_in_memory(
            database, database_labels, query, query_labels)
        training_report['res_{}_avg_map'.format(split)] = avg_map
        training_report['res_{}_wt_avg_map'.format(split)] = wt_avg_map
        training_report['res_{}_avg_mafp'.format(split)] = avg_mafp
        training_report['res_{}_wt_avg_mafp'.format(split)] = wt_avg_mafp


def train_neural_net(working_dir_path, args, data, training_report):
    print('Training a Neural Network model...')
    # Construct network architecture
    ngpus = args.ngpus
    gene_names = data.get_gene_names()
    input_dim, output_dim = data.get_in_out_dims()
    if ngpus > 1:
        import tensorflow as tf
        with tf.device('/cpu:0'):
            template_model, embed_dims = get_model_architecture(
                working_dir_path, args, input_dim, output_dim, gene_names)
        model = multi_gpu_model(template_model, gpus=ngpus)
    else:
        template_model, embed_dims = get_model_architecture(
            working_dir_path, args, input_dim, output_dim, gene_names)
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
    if not args.siamese and not args.triplet:
        plot_accuracy_history(history, join(working_dir_path, 'accuracy.png'))
    print('Evaluating')
    evaluate_model(model, args, data, training_report)
    # Finally, save the model
    save_neural_net(working_dir_path, args, template_model)


def report_config(args, training_report):
    # Data normalization
    if args.sn:
        training_report['cfg_normalization'] = 'sn'
    elif args.gn:
        training_report['cfg_normalization'] = 'gn'
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
    if args.with_dense > 0:
        training_report['cfg_with_dense'] = args.with_dense
    if args.freeze:
        training_report['cfg_freeze_n'] = args.freeze
    training_report['cfg_init'] = 'random'
    if args.init:
        training_report['cfg_init'] = args.init
    # Optimizer
    if args.opt == 'sgd':
        training_report['cfg_opt'] = 'sgd'
        training_report['cfg_lr'] = args.sgd_lr
        training_report['cfg_decay'] = args.sgd_d
        training_report['cfg_momentum'] = args.sgd_m
        training_report['cfg_nesterov?'] = 'Y' if args.sgd_nesterov else 'N'
    if args.early_stop_pat >= 0:
        training_report['cfg_early_stop_patience'] = args.early_stop_pat
        training_report['cfg_early_stop_metric'] = args.early_stop
    if args.checkpoints:
        training_report.cfg_checkpoints = args.checkpoints
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
    data = DataContainer(
        join(
            args.data,
            'train_data.h5'),
        sample_normalize=args.sn,
        feature_normalize=args.gn)
    data.add_split(join(args.data, 'valid_data.h5'), 'valid')
    data.add_split(join(args.data, 'test_data.h5'), 'test')
    if args.gn:
        # save the training data mean and std for later use on new data
        data.mean.to_pickle(join(working_dir, 'mean.p'))
        data.std.to_pickle(join(working_dir, 'std.p'))
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
        train_pca_model(working_dir_path, args, data)
        training_report['cfg_DIMS'] = args.pca
    elif args.nn:
        if args.layerwise_pt:
            layerwise_train_neural_net(
                working_dir_path, args, data, training_report)
        else:
            train_neural_net(working_dir_path, args, data, training_report)

    # Report the configuration and performance of the model
    with open(join(working_dir_path, 'config_results.csv'), 'w') as f:
        for i, col in enumerate(training_report.keys()):
            if i == len(training_report) - 1:
                f.write('{}\n'.format(col))
            else:
                f.write('{},'.format(col))
        for i, val in enumerate(training_report.values()):
            if i == len(training_report) - 1:
                f.write('{}\n'.format(val))
            else:
                f.write('{},'.format(val))
