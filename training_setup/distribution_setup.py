import math
import jax
from jax import numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from distributions.Q_targets import Q_BEC, NoisyQ, Q_Binomial, Q_Coherent, Q_CatState, Q_GKP, Q_Num, Q_PureState, Q_Rotation, Q_Tensor_Product, Q_WState
from distributions.W_targets import W_GKP, NoisyW, W_Binomial, W_CatState, W_Num, W_Rotation, W_Tensor_Product, W_n_particle

"""
single_well_problems = {
    "10" : {
        "name" : "_10",
        "Q" : distribution = Q_BEC(
            phi = jnp.array([1]),
            n=10, 
            num_wells=1,
            normalized=True,
        ),
        "W" : W_n_particle(10),
    },

    "2" : {
        "name" : "_2",
        "Q" : distribution = Q_BEC(
            phi = jnp.array([1]),
            n=2, 
            num_wells=1,
            normalized=True,
        ),
        "W" : W_n_particle(2),
    },

    "coherent" : {
        "name" : "_coherent",
        "Q" : Q_Coherent(
            shift = jnp.array([4,0]),
        ),
        "W" : None,
    },

    "cat" : {
        "name" : "_cat",
        "Q" : Q_CatState(1,2),
        "W" : W_CatState(-1,2,2),
    },

    "pure" : {
        "name" : "_pure",
        "Q" : Q_PureState(jnp.array([0,0,0,0,1+0j])),
        "W" : None,
    },

    "GKP" : {
        "name" : "_GKP",
        "Q" : Q_GKP(0.3,0,20,30),
}
"""


def distribution_name_init(filename, kwargs):
    problem = kwargs["problem"]

    if kwargs["symmetric"] == "True":
        symmetric = True
    else:
        symmetric = False

    if problem == "BH2":
        filename += "_BH2"

    elif problem == "BH3":
        if symmetric:
            filename += "_BH3_symmetric"
        else:
            filename += "_BH3_antisymmetric"

    elif problem == "WState":
        filename += "_WState"

    elif problem == "Test":
        filename += "_Test"

    elif problem == "Multi_Num":
        filename += f"_{kwargs['num_wells']}_Num"

    elif problem == "all":
        filename += "_all"

    elif problem == "all2":
        filename += "_all2"
    
    elif problem == "all3":
        filename += "_all3"
    
    elif problem == "all4":
        filename += "_all4"

    elif problem == "10":
        filename += "_10"

    elif problem == "2":
        filename += "_2"

    elif problem == "coherent":
        filename += "_coherent"

    elif problem == "cat":
        filename += "_cat"

    elif problem == "pure":
        filename += "_pure"

    elif problem == "GKP":
        filename += "_GKP"

    elif problem == "num_0":
        filename += "_num_0"

    elif problem == "num_1":
        filename += "_num_1"

    elif problem == "binom_0":
        filename += "_binom_0"

    elif problem == "binom_1":
        filename += "_binom_1"

    elif problem == "5_W":
        filename += "_5_W"
    
    elif problem == "10_W":
        filename += "_10_W"

    if kwargs["noise"] != 0:
        filename += f"_noise={kwargs['noise']:0.2f}"

    return filename


def Q_distribution_init(kwargs):
    problem = kwargs["problem"]

    if kwargs["symmetric"] == "True":
        symmetric = True
    else:
        symmetric = False
    
    plot_range = [(-7,7,0.3),(-7,7,0.3)]

    if problem == "BH2":
        distribution = Q_BEC(
            phi = jnp.array([1,-1])/jnp.sqrt(2),
            n=100,
            num_wells=2,
            normalized=True,
        )

        plot_range = [(-10,10,0.5),(-10,10,0.5)]

    elif problem == "BH3":

        if symmetric:
            distribution = Q_BEC(
                phi = jnp.array([1,0,1])/jnp.sqrt(2),
                n=60, 
                num_wells=3,
                normalized=True,
            )

        else:
            distribution = Q_BEC(
                phi = jnp.array([1,0,-1])/jnp.sqrt(2),
                n=60,
                num_wells=3,
                normalized=True,
            )

        plot_range = [(-7,7,0.3),(-7,7,0.3)]

    elif problem == "WState":
        distribution = Q_WState(3)

        plot_range = [(-3,3,0.2),(-3,3,0.2)]

    elif problem == "Test":
        distribution1 = Q_CatState(1,2)
        distribution2 = Q_Num(0)

        distribution = Q_Tensor_Product([distribution1, distribution2])

        plot_range = [(-5,5,0.3),(-3,3,0.2)]

    elif problem == "Multi_Num":
        distribution = Q_Tensor_Product([Q_Num(0)]*kwargs["num_wells"])

        plot_range = [(-5,5,60),(-3,3,60)]

    elif problem == "all":
        distribution1 = Q_CatState(1,2)
        distribution2 = Q_BEC(
            phi = jnp.array([1]),
            n=10, 
            num_wells=1,
            normalized=True,
        )
        distribution3 = Q_Num(0)
        distribution4 = Q_Binomial(5,2,mu = 0)
        distribution5 = Q_GKP(0.3,0,20,32)

        distribution = Q_Tensor_Product([distribution1, distribution2, distribution3, distribution4, distribution5])

        plot_range = [(-5,5,50),(-5,5,50)]

    elif problem == "all2":
        distribution1 = Q_CatState(1,2)
        distribution2 = Q_BEC(
            phi = jnp.array([1]),
            n=10, 
            num_wells=1,
            normalized=True,
        )
        distribution3 = Q_Num(0)
        distribution4 = Q_Binomial(5,2,mu = 0)

        distribution5 = Q_CatState(1,1)
        distribution6 = Q_BEC(
            phi = jnp.array([1]),
            n=5, 
            num_wells=1,
            normalized=True,
        )
        distribution7 = Q_Num(1)
        distribution8 = Q_Binomial(5,2,mu = 1)

        distribution = Q_Tensor_Product([distribution1, distribution2, distribution3, distribution4, distribution5, distribution6, distribution7, distribution8])

        plot_range = [(-5,5,50),(-5,5,50)]

    elif problem == "all3":
        distribution1 = Q_CatState(1,1.5)
        distribution2 = Q_BEC(
            phi = jnp.array([1]),
            n=2, 
            num_wells=1,
            normalized=True,
        )
        distribution3 = Q_Num(0)

        distribution4 = Q_Rotation(Q_CatState(1,1), rotation = jnp.array([[0,1],[1,0]]))
        distribution5 = Q_Num(1)

        distribution = Q_Tensor_Product([distribution1, distribution2, distribution3, distribution4, distribution5])

        plot_range = [(-5,5,50),(-5,5,50)]

    elif problem == "all4":
        distribution1 = Q_CatState(1,1.5)
        distribution2 = Q_BEC(
            phi = jnp.array([1]),
            n=2, 
            num_wells=1,
            normalized=True,
        )
        distribution3 = Q_Num(0)

        distribution4 = Q_Rotation(Q_CatState(1,1), rotation = jnp.array([[0,1],[1,0]]))
        distribution5 = Q_Num(1)

        distribution6 = Q_Rotation(Q_CatState(1,1.5), rotation = jnp.array([[0,1],[1,0]]))
        distribution7 = Q_BEC(
            phi = jnp.array([1]),
            n=1.5, 
            num_wells=1,
            normalized=True,
        )
        distribution8 = Q_Rotation(Q_Num(0), rotation = jnp.array([[0,1],[1,0]]))

        distribution9 = Q_CatState(1,1)
        distribution10 = Q_Rotation(Q_Num(1), rotation = jnp.array([[0,1],[1,0]]))

        distribution = Q_Tensor_Product([
            distribution1, 
            distribution2, 
            distribution3, 
            distribution4, 
            distribution5,
            distribution6,
            distribution7,
            distribution8,
            distribution9,
            distribution10,
        ])

        plot_range = [(-5,5,50),(-5,5,50)]

    elif problem == "10":
        distribution = Q_BEC(
            phi = jnp.array([1]),
            n=10, 
            num_wells=1,
            normalized=True,
        )

        plot_range = [(-5,5,50),(-5,5,50)]

    elif problem == "2":
        distribution = Q_BEC(
            phi = jnp.array([1]),
            n=2, 
            num_wells=1,
            normalized=True,
        )

        plot_range = [(-3,3,0.2),(-3,3,0.2)]

    elif problem == "coherent":
        distribution = Q_Coherent(
            shift = jnp.array([4,0]),
        )

        plot_range = [(2,4,0.2),(-2,2,0.2)]

    elif problem == "cat":
        distribution = Q_CatState(1,2)

        plot_range = [(-5,5,50),(-5,5,50)]

    elif problem == "pure":
        distribution = Q_PureState(jnp.array([0,0,0,0,1+0j]))

        plot_range = [(-3,3,0.2),(-3,3,0.2)]

    elif problem == "GKP":
        #init = GKP(0.4,0,20,30)
        distribution = Q_GKP(0.3,0,20,32)

        plot_range = [(-5,5,50),(-5,5,50)]

    elif problem == "num_0":
        distribution = Q_Num(0)

        plot_range = [(-5,5,50),(-5,5,50)]
    
    elif problem == "num_1":
        distribution = Q_Num(1)

        plot_range = [(-5,5,10),(-5,5,10)]

    elif problem == "binom_0":
        distribution = Q_Binomial(5,2,mu = 0)

        plot_range = [(-5,5,50),(-5,5,50)]

    elif problem == "binom_1":
        distribution = Q_Binomial(5,2,mu = 1)

        plot_range = [(-7,7,0.3),(-7,7,0.3)]

    if kwargs["noise"] != 0:
        distribution = NoisyQ(distribution, kwargs["noise"])

    return distribution, plot_range


def W_distribution_init(kwargs):
    problem = kwargs["problem"]

    plot_range = [(-7,7,0.1),(-7,7,0.1)]

    if problem == "BH2":
        raise Exception("BH2 Wigner not yet implemented")

    elif problem == "BH3":
        raise Exception("BH3 Wigner not yet implemented")

    elif problem == "10":
        distribution = W_n_particle(10)

        plot_range = [(-7,7,0.1),(-7,7,0.1)]

    elif problem == "coherent":
        raise Exception("Coherent Wigner not yet implemented")

    elif problem == "cat":
        distribution = W_CatState(-1,2,2)

        plot_range = [(-4,4,50),(-2,2,50)]

    elif problem == "pure":
        raise Exception("Pure state not yet implemented")

    elif problem == "GKP":
        distribution = W_GKP(0.3,0,20,30)

        plot_range = [(-3,3,0.1),(-3,3,0.1)]

    elif problem == "num_0":
        distribution = W_Num(0)

        plot_range = [(-5,5,0.1),(-5,5,0.1)]
    
    elif problem == "num_1":
        distribution = W_Num(1)

        plot_range = [(-5,5,0.1),(-5,5,0.1)]

    elif problem == "binom_0":
        distribution = W_Binomial(5,2,mu = 0)

        plot_range = [(-7,7,0.1),(-7,7,0.1)]

    elif problem == "binom_1":
        distribution = W_Binomial(5,2,mu = 1)

        plot_range = [(-7,7,0.1),(-7,7,0.1)]

    elif problem == "10_W":

        distribution3 = W_Num(0)

        distribution = W_Tensor_Product([
            distribution3,
            distribution3,
            distribution3
            #distribution3
            # distribution3
        ])

        plot_range = [(-5,5,0.1),(-5,5,0.1)]

    elif problem == "5_W":

        theta_swap = jnp.array([[0, 1], [1, 0]])
        theta = -1 * (jnp.pi / 2)

        rotation_45_cw = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta),  jnp.cos(theta)],
        ])

        distribution1 = W_Rotation(W_CatState(1, 1.5, 2), rotation=theta_swap)
        distribution2 = W_n_particle(1)
        distribution3 = W_Num(0)
        distribution4 = W_CatState(1, 1, 2)
        distribution5 = W_n_particle(2)

        distribution = W_Tensor_Product([
            distribution1,
            distribution2,
            distribution3,
            distribution4,
            #distribution5,
        ])

    elif problem == "all3":

        distribution1 = W_CatState(1,1.5,2)
        distribution2 = W_n_particle(2)

        # theta = -1 * (2 * jnp.pi / 4)

        # rotation_45_cw = jnp.array([
        #     [jnp.cos(theta), -jnp.sin(theta)],
        #     [jnp.sin(theta),  jnp.cos(theta)],
        # ])

        distribution3 = W_Num(0)
        
        #W_Rotation(W_Num(0), rotation=rotation_45_cw)

        # distribution4 = W_Rotation(W_CatState(1,1,2), rotation = jnp.array([[0,1],[1,0]]))

        # distribution5 = W_Num(1)

        distribution = W_Tensor_Product([distribution1, distribution2, distribution3])

        plot_range = [(-5,5,0.1),(-5,5,0.1)]

    if problem == "QST_CGAN_W_Neg":
        distribution = None

    if kwargs["noise"] != 0:
        distribution = NoisyW(distribution, kwargs["noise"])

    return distribution, plot_range
