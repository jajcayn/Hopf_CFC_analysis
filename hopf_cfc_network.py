"""
Hopf oscillators for studying CFCs.

neurolib params vs. Qin et al. paper
neurolib ||  Qin    || meaning
    `a`  || `sigma` || bifurcation parameter
    `w`  || `omega` || frequency
"""

import numpy as np
from neurolib.models.multimodel.builder import HopfNode, Network
from neurolib.models.multimodel.builder.hopf import HOPF_DEFAULT_PARAMS
from neurolib.utils.stimulus import ZeroInput
from scipy.linalg import block_diag

# makes intrinsic frequency ~ 6Hz
SLOW_HOPF_DEFAULT_PARAMS = {**HOPF_DEFAULT_PARAMS, "w": 0.04}
# makes intrinsic frequency ~ 30Hz
FAST_HOPF_DEFAULT_PARAMS = {**HOPF_DEFAULT_PARAMS, "w": 0.2}


class SlowHopfNode(HopfNode):
    """
    Little helper for slow Hopf oscillator.
    """

    name = "Slow Hopf node"
    label = "SlowHopf"

    def __init__(self, params=SLOW_HOPF_DEFAULT_PARAMS, seed=None):
        super().__init__(params=params, seed=seed)


class FastHopfNode(HopfNode):
    """
    Little helper for fast Hopf oscillator.
    """

    name = "Fast Hopf node"
    label = "FastHopf"

    def __init__(self, params=FAST_HOPF_DEFAULT_PARAMS, seed=None):
        super().__init__(params=params, seed=seed)


class SlowFastHopfNetwork(Network):
    """
    Simplified brain network of Hopf oscillators where a single "unit" is made
    of one "slow" and one "fast" Hopf oscillators.
    """

    name = "Slow-fast Hopf network"
    label = "SlowFastHopfNet"

    sync_variables = ["network_x", "network_y"]
    # additive coupling, like in the Qin et al. paper, i.e. only linearly added
    # activity from other regions, scaled by a constant (the coupling strength)
    default_coupling = {
        "network_x": "multiplicative",
        "network_y": "multiplicative",
    }
    output_vars = ["x", "y"]

    def __init__(
        self,
        number_of_slow_fast_units=1,
        slow_to_fast_connection=5.0,
        fast_to_slow_connection=0.0,
        slow_to_slow_connection=None,
    ):
        # when we have more basic units, assert there is slow<->slow connection
        if number_of_slow_fast_units > 1:
            assert slow_to_slow_connection is not None
        # build connectivity matrix
        # we always start with single unit as [slow, fast], i.e. for two unit
        # systems the order is [slow1, fast1, slow2, fast2]
        # all matrices are in the format [to, from]
        single_unit_matrix = np.array(
            [[0.0, fast_to_slow_connection], [slow_to_fast_connection, 0.0]]
        )
        # now we create block-diagonal matrix with `single_unit_matrix` on the
        # diagonal - those are within-unit connections
        connectivity_matrix = block_diag(
            *[single_unit_matrix] * number_of_slow_fast_units
        )
        # now create indices in the full matrix where we have slow<->slow
        slow_slow_ndxs = [
            (i, j)
            for i in range(0, number_of_slow_fast_units * 2, 2)
            for j in range(0, number_of_slow_fast_units * 2, 2)
            if i != j
        ]
        # and we set the slow<->slow connection for all slow<->slow interactions
        for ni, nj in slow_slow_ndxs:
            connectivity_matrix[ni, nj] = slow_to_slow_connection
            connectivity_matrix[nj, ni] = slow_to_slow_connection

        # set up nodes
        nodes = []
        for i in range(number_of_slow_fast_units):
            # init with default parameters
            node_slow = SlowHopfNode(params=SLOW_HOPF_DEFAULT_PARAMS)
            node_fast = FastHopfNode(params=FAST_HOPF_DEFAULT_PARAMS)
            # set correct index - slow even, fast odd
            node_slow.index = i * 2
            node_fast.index = i * 2 + 1
            # set index of state variables
            node_slow.idx_state_var = i * (
                node_slow.num_state_variables + node_fast.num_state_variables
            )
            node_fast.idx_state_var = (
                node_slow.idx_state_var + node_slow.num_state_variables
            )
            # default noise is none
            node_slow.noise_input = [
                ZeroInput()
            ] * node_slow.num_noise_variables
            node_fast.noise_input = [
                ZeroInput()
            ] * node_fast.num_noise_variables
            nodes += [node_slow, node_fast]

        super().__init__(
            nodes=nodes,
            connectivity_matrix=connectivity_matrix,
            # by default no delay
            delay_matrix=np.zeros_like(connectivity_matrix),
        )
        # get all coupling variables
        all_couplings = [
            mass.coupling_variables
            for node in self.nodes
            for mass in node.masses
        ]
        # assert they are the same
        assert all(all_couplings[0] == coupling for coupling in all_couplings)
        # invert as to name: idx
        self.coupling_symbols = {v: k for k, v in all_couplings[0].items()}

        # TODO within-unit connections multiplicative, between-unit connections
        # TODO diffusive
