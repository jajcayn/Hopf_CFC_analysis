"""
Run exploration with medium resolution within the connection strengths (15x15), but with 3x3 frequencies for the oscillators
and 5x5 bifurcation parameters.

"""

import numpy as np
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.functions import getPowerSpectrum
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.models.multimodel import MultiModel
from hopf_cfc_network import SlowFastHopfNetwork
from cfcUtils import modulation_index_general, mean_vector_length, phase_locking_value, mutual_information
from scipy.signal import find_peaks



# w intrinsic
# 0.003 ~ 0.5Hz
# 0.005 ~ 0.8Hz
# 0.01 ~ 2Hz
# 0.05 ~ 8Hz
# 0.06 ~ 10Hz
# 0.08 ~ 12Hz
# 0.1 ~ 17Hz
# 0.2 ~ 32Hz
# 0.3 ~ 50Hz

DURATION = 20.0 * 1000  # ms
DT = 0.1  # ms

model = MultiModel(
    SlowFastHopfNetwork(
        number_of_slow_fast_units=1,
        slow_to_fast_connection=0.0,
        fast_to_slow_connection=0.0,
    )
)

model.params["duration"] = DURATION
model.params["sampling_dt"] = DT
model.params["dt"] = 0.01  # ms - lower dt for numba backend
# numba backend is the way to go in exploration - much faster
model.params["backend"] = "numba"

# manually add params you want to change during exploration, btw model.params is just a dictionary, so you can add whatever :)
model.params["slow_to_fast"] = 0.0
model.params["fast_to_slow"] = 0.0
model.params["bifurcation_param_slow"] = 0.25
model.params["bifurcation_param_fast"] = 0.25
model.params["frequency_slow"] = 0.04
model.params["frequency_fast"] = 0.2



parameters = ParameterSpace(
    {
        "slow_to_fast": np.linspace(0.0, 2., 15),
        "fast_to_slow": np.linspace(0.0, 2., 15),
        "frequency_slow": np.array([0.02, 0.04, 0.08]),
        "frequency_fast": np.array([0.15, 0.25, 0.35]),
        "bifurcation_param_slow": np.array([0.25, 0.6, 1.3, 4, 8]),
        "bifurcation_param_fast": np.array([0.25, 0.4, 0.6, 1, 4]),
    },
    allow_star_notation=True,
    kind="grid",
)

# Default params:
# slow_frequency: 0.04
# fast_frequency: 0.2
# bifurcation param: 0.25


def evaluateSimulation(traj):
    # get model with parameters for this run
    model = search.getModelFromTraj(traj)
    # extract stuff you want
    s_f_conn = model.params["slow_to_fast"]
    f_s_conn = model.params["fast_to_slow"]

    model.params["*connectivity"] = np.array([[0.0, f_s_conn], [s_f_conn, 0.0]])
    model.params['SlowFastHopfNet.SlowHopf_0.HopfMass_0.a'] = model.params["bifurcation_param_slow"]
    model.params['SlowFastHopfNet.FastHopf_1.HopfMass_0.a'] = model.params["bifurcation_param_fast"]
    model.params['SlowFastHopfNet.SlowHopf_0.HopfMass_0.w'] = model.params["frequency_slow"]
    model.params['SlowFastHopfNet.FastHopf_1.HopfMass_0.w'] = model.params["frequency_fast"]

    model.run()


    phase_slow = np.arctan2(model.y[0, :], model.x[0, :])
    phase_fast = np.arctan2(model.y[1, :], model.x[1, :])
    amp_fast = np.sqrt(model.x[1, :] ** 2 + model.y[1, :] ** 2)

    mi = modulation_index_general(amp_fast, phase_slow, n_bins = 18)
    mvl = mean_vector_length(amp_fast, phase_slow)
    plv = phase_locking_value(phase_fast, phase_slow)
    minfo = mutual_information(phase_fast, phase_slow, bins=16, log2=False)

    freq_slow, pow_slow = getPowerSpectrum(model.x.T[:, 0], dt=0.1, maxfr=60, spectrum_windowsize=1)
    freq_fast, pow_fast = getPowerSpectrum(model.x.T[:, 1], dt=0.1, maxfr=60, spectrum_windowsize=1)

    peaks_fast, _ = find_peaks(pow_fast, height=max(1e-3, 1.0 * np.std(pow_fast)))
    peaks_slow, _ = find_peaks(pow_slow, height=max(1e-3, 0.5 * np.std(pow_slow)))



    result_dict = {
        "peaks_freq_fast": peaks_fast,
        "peaks_freq_slow": peaks_slow,
        "modulation_index": mi,
        "mean_vector_length": mvl,
        "phase_locking_value": plv,
        "mutual_information": minfo,
    }


    search.saveToPypet(result_dict, traj)


search = BoxSearch(
    model=model,
    evalFunction=evaluateSimulation,
    parameterSpace=parameters,
    filename="/mnt/raid/data/laura/Hopf_CFC_results/medium_resolution_exploration.hdf",
    ncores=50,
)



search.run()
