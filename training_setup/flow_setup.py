import jax
from jax import random
jax.config.update("jax_enable_x64", True)

from Q_flows.convex_potential import get_CPF
from Q_flows.flow_util import FFJORDNet, ResNet, flow_mixture, sine
from distributions.W_function import QWigner

from distributions.Q_function import QFlow
from Q_flows.affine import get_realNVP
from Q_flows.continuous import get_FFJORD

def flow_name_init(filename, kwargs):
    model = kwargs["model"]

    if model == "CPF":
        filename = "CPF" + filename

    elif model == "CNF":
        filename = "CNF" + filename
        
        if kwargs["network"] != "ResNet":
            filename = filename + "_FFJORD"

        if kwargs["activation"] != "GELU":
            filename = filename + f"_{kwargs['activation']}"

        if kwargs["step_controller"] == "Constant":
            filename = filename + f"_Step={kwargs['step_size']}"
        else:
            filename = filename + f"_Adaptive"

    elif model == "RealNVP":
        filename = "RealNVP" + filename

    elif model == "RQS":
        filename = "RQS" + filename

    if kwargs["layer_size"] != 20:
        filename += f"_layer_size={kwargs['layer_size']}"
    
    if kwargs["network_layers"] != 3:
        filename += f"_network_layers={kwargs['network_layers']}"

    if kwargs["num_encodings"] != 0:
        filename += f"_num_encodings={kwargs['num_encodings']}"

    if kwargs["ensemble_number"] != 1:
        filename = filename + f"_ensemble={kwargs['ensemble_number']}"
    
    return filename

def flow_init(kwargs, rescale):
    model = kwargs["model"]

    if model == "CPF":
        flow = get_CPF(
            hidden_layer_size=kwargs["layer_size"], 
            augmented_layer_size=4, 
            num_layers=kwargs["network_layers"], 
            scale_layer = True,
        )
        
    elif model == "CNF":
        flow = get_FFJORD(
            num_layers = kwargs["network_layers"],
            internal_layer_size = kwargs["layer_size"],
            scale_layer=True,
            adaptive = kwargs["step_controller"] == "Adaptive",
            dt = kwargs["step_size"],
            model = ResNet if kwargs["network"] == "ResNet" else FFJORDNet,
            combine_ty = True if kwargs["network"] == "ResNet" else False,
            activation = jax.nn.gelu if kwargs["activation"] == "GELU" else sine,
            num_encodings = kwargs["num_encodings"],
            rescale = rescale,
        )

    elif model == "RealNVP":
        flow = get_realNVP(
            num_layers = 3,
            num_transform_layers = kwargs["network_layers"],
            internal_layer_size = kwargs["layer_size"],
            random_init = True,
            divide = 10000,
            shift = True,
            random_shift = True,
        )

    elif model == "RQS":
        flow = get_realNVP(
            num_layers = 2,
            num_transform_layers = kwargs["network_layers"],
            internal_layer_size = kwargs["layer_size"],
            coupling_layer = "RQS",
            knots = 5,
            B = 3,
            #permutation_layer = "linear"
        )

    if kwargs["ensemble_number"] != 1:
        flow = flow_mixture(flow, kwargs["ensemble_number"])

    return flow

def Q_flow_setup(flow, init_samples):
    model = QFlow(
        flow_init = flow,
        input_dim = init_samples.shape[1],
    )

    params = model.init(
        rng = random.PRNGKey(0),
        inputs = init_samples,
        real_data = True
    )

    return model, params

def W_flow_setup(flow, init_samples, kwargs):
    model = QWigner(
        flow_init = flow,
        input_dim = init_samples.shape[1],
        positive_scale = True if kwargs["positive_scale"]=="True" else False,
    )

    params = model.init(
        rng = random.PRNGKey(0),
        inputs = init_samples,
        real_data = True,
    )

    return model, params