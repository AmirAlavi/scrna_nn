from keras.models import load_model

from .sparse_layer import Sparse

def get_pt_model(pt_base_file_name, args):
    model_file = pt_base_file_name
    if args['--sn']:
        model_file += "_sn"
    elif args['--gn']:
        model_file += "_gn"
    model_file += "_" + + args['--act'] + '.h5'
    return load_model(model_file, custom_objects={'Sparse': Sparse})

def set_pt_GO(model, args):
    first_lvl_pt_file = "../pre_trained/GO_first_lvl/model"
    second_lvl_pt_file = "../pre_trained/GO_second_lvl/model"
    third_lvl_pt_file = "../pre_trained/GO_third_lvl/model"
    dense_pt_file = "../pre_trained/dense_31/model"
    
    first_lvl_pt_model = get_pt_model(first_lvl_pt_file, args)
    first_lvl_pt_wts = first_lvl_pt_model.layers[-1].get_weights()[1:]
    second_lvl_pt_model = get_pt_model(second_lvl_pt_file, args)
    second_lvl_pt_wts = second_lvl_pt_model.layers[-1].get_weights()[1:]
    third_lvl_pt_model = get_pt_model(third_lvl_pt_file, args)
    third_lvl_pt_wts = third_lvl_pt_model.layers[-1].get_weights()[1:]
    dense_pt_model = get_pt_model(dense_pt_file, args)
    dense_pt_wts = dense_pt_model.layers[-1].get_weights()[1:]
    
    model.layers[1].set_weights(first_lvl_pt_wts)
    model.layers[2].set_weights(second_lvl_pt_wts)
    model.layers[3].set_weights(third_lvl_pt_wts)
    model.layers[4].set_weights(dense_pt_wts)
    print("loaded pretrained weights for GO lvls model")
    
def set_pt_flatGO(model, args):
    flatGO_pt_file = "../pre_trained/flatGO/model"
    dense_pt_file = "../pre_trained/dense_100/model"
    
    flatGO_pt_model = get_pt_model(first_lvl_pt_file, args)
    first_lvl_pt_wts = first_lvl_pt_model.layers[-1].get_weights()[1:]
    dense_pt_model = get_pt_model(dense_pt_file, args)
    dense_pt_wts = dense_pt_model.layers[-1].get_weights()[1:]
    
    model.layers[1].set_weights(first_lvl_pt_wts)
    model.layers[2].set_weights(second_lvl_pt_wts)
    model.layers[3].set_weights(third_lvl_pt_wts)
    model.layers[4].set_weights(dense_pt_wts)
    print("loaded pretrained weights for GO lvls model")
    
def set_pretrained_weights(model, model_name, args):
    if model_name == 'GO':
        set_pt_GO(model, args)
