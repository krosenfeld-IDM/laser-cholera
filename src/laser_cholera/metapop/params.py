import gzip
import io
import json
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Optional
from typing import Union

import click
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure

from laser_cholera.sc import printcyan
from laser_cholera.sc import printred


def get_parameters(filename: Optional[Union[str, Path]] = None, overrides: Optional[dict] = None) -> PropertySet:
    fn_map = {
        (".json",): load_json_parameters,
        (".json", ".gz"): load_compressed_json_parameters,
        (".h5",): load_hdf5_parameters,
        (".hdf",): load_hdf5_parameters,
        (".hdf5",): load_hdf5_parameters,
        (".h5", ".gz"): load_compressed_hdf5_parameters,
        (".hdf", ".gz"): load_compressed_hdf5_parameters,
        (".hdf5", ".gz"): load_compressed_hdf5_parameters,
    }

    file_path = Path(filename) if filename is not None else Path(__file__).parent / "data" / "default_parameters.json"
    suffixes = [suffix.lower() for suffix in file_path.suffixes]
    load_fn = fn_map[tuple(suffixes)]

    params = load_fn(file_path)

    validate(params)

    if "verbose" not in params:
        params.verbose = False

    if overrides is not None:
        params += overrides

        if params.verbose:
            click.echo(f"Updated/overrode file parameters with `{overrides}`…")

    if params.verbose:
        click.echo(f"Loaded parameters from `{filename}`…")

    return params


def load_json_parameters(filename: Union[str, Path]) -> PropertySet:
    file_path = Path(filename)
    with file_path.open("r") as file:
        parameters = json.load(file)

    return sanitize_json(parameters)


def load_compressed_json_parameters(filename: Union[str, Path]) -> PropertySet:
    file_path = Path(filename)
    with gzip.open(file_path, "rb") as gz_file:
        with io.BytesIO(gz_file.read()) as file:
            parameters = json.load(file)

    return sanitize_json(parameters)


def sanitize_json(parameters: dict) -> PropertySet:
    # Note the following canonicalizes the order of the locations based on the
    # order in the JSON file.
    # We might consider either
    # a) alphabetical order by location name or
    # b) order by int(ID)

    params = PropertySet(parameters)

    params.date_start = datetime.strptime(params.date_start, "%Y-%m-%d")  # noqa: DTZ007
    params.date_stop = datetime.strptime(params.date_stop, "%Y-%m-%d")  # noqa: DTZ007
    params.nticks = (params.date_stop - params.date_start).days + 1
    printcyan(f"Simulation calendar dates: {params.date_start} to {params.date_stop} ({params.nticks} ticks)")

    nticks = params.nticks
    npatches = len(params.location_id)

    # IDs are 1-based not 0-based like indices
    # Map string IDs to names and names to indices (0-based)

    strid_to_location_map = {str(id): name for id, name in zip(parameters["location_id"], parameters["location_name"])}
    location_to_idx_map = {name: idx for idx, name in enumerate(sorted(parameters["location_name"]))}
    strid_to_numidx_map = {id: location_to_idx_map[name] for id, name in strid_to_location_map.items()}

    params.location_id = np.array(params.location_id, dtype=np.uint32)
    printcyan(f"{len(params.location_id)=}")
    printcyan(f"{len(params.location_name)=}")

    params.N_j_initial = np.array(params.N_j_initial, dtype=np.uint32)
    params.S_j_initial = np.array(params.S_j_initial, dtype=np.uint32)
    params.E_j_initial = np.array(params.E_j_initial, dtype=np.uint32)
    params.I_j_initial = np.array(params.I_j_initial, dtype=np.uint32)
    params.R_j_initial = np.array(params.R_j_initial, dtype=np.uint32)
    params.V1_j_initial = np.array(params.V1_j_initial, dtype=np.uint32)
    params.V2_j_initial = np.array(params.V2_j_initial, dtype=np.uint32)

    b_jt = np.zeros((nticks, npatches), dtype=np.float32)
    for key, values in params.b_jt.items():
        b_jt[:, strid_to_numidx_map[key]] = values
    params.b_jt = b_jt

    d_jt = np.zeros((nticks, npatches), dtype=np.float32)
    for key, values in params.d_jt.items():
        d_jt[:, strid_to_numidx_map[key]] = values
    params.d_jt = d_jt

    # Transpose here for more efficient access by tick
    nu_1_jt = np.zeros((nticks, npatches), dtype=np.float32)
    for key, values in params.nu_1_jt.items():
        nu_1_jt[:, strid_to_numidx_map[key]] = values
    params.nu_1_jt = nu_1_jt  # Change from dict to np.ndarray

    # Transpose here for more efficient access by tick
    nu_2_jt = np.zeros((nticks, npatches), dtype=np.float32)
    for key, values in params.nu_2_jt.items():
        nu_2_jt[:, strid_to_numidx_map[key]] = values
    params.nu_2_jt = nu_2_jt  # Change from dict to np.ndarray

    mu_jt = np.zeros((nticks, npatches), dtype=np.float32)
    for key, values in params.mu_jt.items():
        mu_jt[:, strid_to_numidx_map[key]] = values
    params.mu_jt = mu_jt

    params.beta_j0_hum = np.array(params.beta_j0_hum, dtype=np.float32)

    ndays = 366
    beta_j_seasonality = np.zeros((ndays, npatches), dtype=np.float32)
    for key, values in params.beta_j_seasonality.items():
        beta_j_seasonality[:, strid_to_numidx_map[key]] = values
    params.beta_j_seasonality = beta_j_seasonality

    params.tau_i = np.array(params.tau_i, dtype=np.float32)
    assert np.all((params.tau_i >= 0.0) & (params.tau_i <= 1.0)), "tau_i values must be in the range [0, 1]"

    pi_ij = np.zeros((npatches, npatches), dtype=np.float32)
    for key, values in params.pi_ij.items():
        pi_ij[strid_to_numidx_map[key], : len(values)] = [value if value != "NA" else 0.0 for value in values]
    params.pi_ij = pi_ij

    params.beta_j0_env = np.array(params.beta_j0_env, dtype=np.float32).reshape(-1, 1)
    params.theta_j = np.array(params.theta_j, dtype=np.float32)

    psi_jt = np.zeros((nticks, npatches), dtype=np.float32)
    for key, values in params.psi_jt.items():
        psi_jt[:, strid_to_numidx_map[key]] = values
    params.psi_jt = psi_jt

    # TODO - remove this
    if params.delta_min > params.delta_max:
        printred("Swapping delta_min and delta_max")
        params.delta_min, params.delta_max = params.delta_max, params.delta_min

    return params


def load_hdf5_parameters(filename: Union[str, Path]) -> PropertySet:
    with h5.File(filename, "r") as h5file:
        parameters = load_hdf5(h5file)

    return parameters


def load_compressed_hdf5_parameters(filename: Union[str, Path]) -> PropertySet:
    with gzip.open(filename, "rb") as gz_file:
        with io.BytesIO(gz_file.read()) as file:
            with h5.File(file, "r") as h5file:
                parameters = load_hdf5(h5file)

    return parameters


def load_hdf5(h5file) -> PropertySet:
    ps = PropertySet()

    # date_start and date_stop
    start = h5file["date_start"][()][0]
    stop = h5file["date_stop"][()][0]
    epoch = datetime(year=1970, month=1, day=1)
    ps.date_start = epoch + timedelta(days=start)
    ps.date_stop = epoch + timedelta(days=stop)
    nticks = ps.nticks = (ps.date_stop - ps.date_start).days + 1  # +1 to include stop date

    # scalars
    for scalar in [
        "phi_1",
        "phi_2",
        "omega_1",
        "omega_2",
        "iota",
        "gamma_1",
        "gamma_2",
        "epsilon",
        "rho",
        "sigma",
        "alpha_1",
        "alpha_2",
        "zeta_1",
        "zeta_2",
        "kappa",
        "delta_min",
        "delta_max",
    ]:
        ps[scalar] = h5file[scalar][()][0]

    # TODO - remove this
    if ps.delta_min > ps.delta_max:
        printred("Swapping delta_min and delta_max")
        ps.delta_min, ps.delta_max = ps.delta_max, ps.delta_min

    # per location vectors
    for vector in [
        "location_id",
        "location_name",
        "N_j_initial",
        "S_j_initial",
        "E_j_initial",
        "I_j_initial",
        "R_j_initial",
        "V1_j_initial",
        "V2_j_initial",
        "beta_j0_hum",
        "beta_j0_env",
        "tau_i",
        "theta_j",
    ]:
        ps[vector] = h5file[vector][()]

    npatches = len(ps.location_name)

    # per tick, per location arrays
    for array in ["b_jt", "d_jt", "nu_1_jt", "nu_2_jt", "mu_jt", "psi_jt"]:
        temp = np.zeros((nticks, npatches), dtype=np.float32)
        for patch in ps.location_id:
            ipatch = int(patch) - 1  # 1-based to 0-based
            temp[:, ipatch] = h5file[array][str(patch)][()]
        ps[array] = temp

    # beta_j_seasonality npatches vectors of 366 entries
    temp = np.zeros((366, npatches), dtype=np.float32)
    for patch in ps.location_id:
        ipatch = int(patch) - 1
        temp[:, ipatch] = h5file["beta_j_seasonality"][str(patch)][()]
    ps["beta_j_seasonality"] = temp

    # pi_ij npatches vectors of npatches entries
    temp = np.zeros((npatches, npatches), dtype=np.float32)
    for patch in ps.location_id:
        ipatch = int(patch) - 1
        # NumPy has a warning about cast NaN to float which horks testing, just copy non-NaN values
        notnan = np.logical_not(np.isnan(h5file["pi_ij"][str(patch)][()]))
        temp[ipatch, notnan] = h5file["pi_ij"][str(patch)][()][notnan]
    # Ensure that the diagonal of temp is either NaN or zero
    assert np.all((np.diag(temp) == 0.0) | np.isnan(np.diag(temp))), "Diagonal values of temp must be either NaN or zero"
    np.fill_diagonal(temp, 0)
    ps["pi_ij"] = temp

    return ps


def validate(params: PropertySet) -> None:
    # date_start and date_stop
    assert params.date_stop >= params.date_start, f"date_stop ({params.date_stop}) must be >= date_start ({params.date_start})"

    # location_id and location_name
    assert len(params.location_id) == len(params.location_name), (
        f"Number of location IDs ({len(params.location_id)}) must match number of location names ({len(params.location_name)})"
    )
    npatches = len(params.location_name)

    # N_j_initial == S_j_initial + E_j_initial + I_j_initial + R_j_initial + V1_j_initial + V2_j_initial
    assert np.all(
        params.N_j_initial
        == params.S_j_initial + params.E_j_initial + params.I_j_initial + params.R_j_initial + params.V1_j_initial + params.V2_j_initial
    ), "Initial populations must sum to N_j_initial"

    assert len(params.N_j_initial) == npatches, (
        f"Number of N_j_initial values ({len(params.N_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.N_j_initial > 0), "N_j_initial values must be positive"
    assert len(params.S_j_initial) == npatches, (
        f"Number of S_j_initial values ({len(params.S_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.S_j_initial >= 0), "S_j_initial values must be non-negative"
    assert len(params.E_j_initial) == npatches, (
        f"Number of E_j_initial values ({len(params.E_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.E_j_initial >= 0), "E_j_initial values must be non-negative"
    assert len(params.I_j_initial) == npatches, (
        f"Number of I_j_initial values ({len(params.I_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.I_j_initial >= 0), "I_j_initial values must be non-negative"
    assert len(params.R_j_initial) == npatches, (
        f"Number of R_j_initial values ({len(params.R_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.R_j_initial >= 0), "R_j_initial values must be non-negative"
    assert len(params.V1_j_initial) == npatches, (
        f"Number of V1_j_initial values ({len(params.V1_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.V1_j_initial >= 0), "V1_j_initial values must be non-negative"
    assert len(params.V2_j_initial) == npatches, (
        f"Number of V2_j_initial values ({len(params.V2_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.V2_j_initial >= 0), "V2_j_initial values must be non-negative"

    nticks = params.nticks

    # shape of b_jt = (nticks, npatches)
    assert params.b_jt.shape == (nticks, npatches), (
        f"Shape of b_jt ({params.b_jt.shape}) does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # 0 <= b_jt <= 0.0001369863 (5% annual / 365 days)
    assert np.all((params.b_jt >= 0.0) & (params.b_jt <= 0.0001369863)), "b_jt values must be in the range [0, 0.0001369863]"

    # shape of b_jt = (nticks, npatches)
    assert params.d_jt.shape == (nticks, npatches), (
        f"Shape of d_jt ({params.d_jt.shape}) does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # 0 <= d_jt <= 0.0002739726 (10% annual / 365 days)
    assert np.all((params.d_jt >= 0.0) & (params.d_jt <= 0.0002739726)), "d_jt values must be in the range [0, 0.0002739726]"

    # shape of nu_1_jt = (nticks, npatches)
    assert params.nu_1_jt.shape == (nticks, npatches), (
        f"Shape of nu_1_jt ({params.nu_1_jt.shape}) does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # nu_1_jt - no daily value can be larger than the country population (N_j_initial) / 7
    assert np.all(params.nu_1_jt <= params.N_j_initial[np.newaxis, :] / 7), (
        "nu_1_jt values must not exceed N_j_initial / 7 for any location"
    )

    # shape of nu_2_jt = (nticks, npatches)
    assert params.nu_2_jt.shape == (nticks, npatches), (
        f"Shape of nu_2_jt ({params.nu_2_jt.shape}) does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # nu_2_jt - no daily value can be larger than the country population (N_j_initial) / 7
    assert np.all(params.nu_2_jt <= params.N_j_initial[np.newaxis, :] / 7), (
        "nu_2_jt values must not exceed N_j_initial / 7 for any location"
    )

    # phi_1 and phi_2 must be between 0 (completely ineffective) and 1 (fully effective)
    assert (params.phi_1 >= 0.0) & (params.phi_1 <= 1.0), "phi_1 value must be in the range [0, 1]"
    assert (params.phi_2 >= 0.0) & (params.phi_2 <= 1.0), "phi_2 value must be in the range [0, 1]"

    # omega_1 and omega_2 must be between 0 (no waning) and 1 (1 day duration)
    assert (params.omega_1 >= 0.0) & (params.omega_1 <= 1.0), "omega_1 value must be in the range [0, 1]"
    assert (params.omega_2 >= 0.0) & (params.omega_2 <= 1.0), "omega_2 value must be in the range [0, 1]"

    # iota must be between 0.002739726 (1/365 days) and 1 (1 day duration)
    assert (params.iota >= 0.002739726) & (params.iota <= 1.0), "iota value must be in the range [0.002739726, 1]"

    # gamma_1 and gamma_2 must be between 0.002739726 (1/365 days) and 1 (1 day duration)
    assert (params.gamma_1 >= 0.002739726) & (params.gamma_1 <= 1.0), "gamma_1 value must be in the range [0.002739726, 1]"
    assert (params.gamma_2 >= 0.002739726) & (params.gamma_2 <= 1.0), "gamma_2 value must be in the range [0.002739726, 1]"

    # epsilon must be between 0 (no waning) and 1 (1 day duration)
    assert (params.epsilon >= 0.0) & (params.epsilon <= 1.0), "epsilon value must be in the range [0, 1]"

    # shape of mu_jt = (nticks, npatches)
    assert params.mu_jt.shape == (nticks, npatches), (
        f"Shape of mu_jt ({params.mu_jt.shape}) does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # all mu_jt must be between 0 (no mortality) and 1 (100% mortality)
    assert np.all((params.mu_jt >= 0.0) & (params.mu_jt <= 1.0)), "mu_jt values must be in the range [0, 1]"

    # rho must be between 0 (all false positives) and 1 (no false positives)
    assert (params.rho >= 0.0) & (params.rho <= 1.0), "rho value must be in the range [0, 1]"

    # sigma must be between 0 (all asymptomatic) and 1 (all symptomatic)
    assert (params.sigma >= 0.0) & (params.sigma <= 1.0), "sigma value must be in the range [0, 1]"

    # length of beta_j0_hum must be equal to number of locations
    assert len(params.beta_j0_hum) == npatches, (
        f"Number of beta_j0_hum values ({len(params.beta_j0_hum)}) does not match number of locations ({npatches})"
    )
    # beta_j0_hum must be >= 0
    assert np.all(params.beta_j0_hum >= 0.0), "beta_j0_hum values must be >= 0"

    # shape of beta_j_seasonality = (ndays, npatches)
    ndays = 366
    assert params.beta_j_seasonality.shape == (ndays, npatches), (
        f"Shape of beta_j_seasonality ({params.beta_j_seasonality.shape}) does not match (ndays, npatches) = ({ndays}, {npatches})"
    )
    # beta_j_seasonality must be >= -b1
    printred("Clipping beta_j_seasonality to >= -1.0", style="bold")
    params.beta_j_seasonality = np.clip(params.beta_j_seasonality, -1.0, None)
    assert np.all(params.beta_j_seasonality >= -1.0), "beta_j_seasonality values must be >= -1.0"

    # length of tau_i must be equal to number of locations
    assert len(params.tau_i) == npatches, f"Number of tau_i values ({len(params.tau_i)}) does not match number of locations ({npatches})"
    # tau_i must be between 0 (no emigration) and 1 (all emigration)
    assert np.all((params.tau_i >= 0.0) & (params.tau_i <= 1.0)), "tau_i values must be in the range [0, 1]"

    # shape of pi_ij = (npatches, npatches)
    assert params.pi_ij.shape == (npatches, npatches), (
        f"Shape of pi_ij ({params.pi_ij.shape}) does not match (npatches, npatches) = ({npatches}, {npatches})"
    )
    # all pi_ij values must be between 0 (no connectivity) and 1 (full connectivity)
    assert np.all((params.pi_ij >= 0.0) & (params.pi_ij <= 1.0)), "pi_ij values must be in the range 0 <= pi_ij <= 1"
    # pi_ij diagonal should be all 0
    assert np.all(np.diag(params.pi_ij) == 0.0), "pi_ij diagonal values must be 0"
    # pi_ij sum of axis 1 must be <= 1
    assert np.all(np.sum(params.pi_ij, axis=1) <= 1.0), "pi_ij row sums must be <= 1"

    # alpha_1 and alpha_2
    # TODO - TBD

    # length of beta_j0_env must be equal to number of locations
    assert len(params.beta_j0_env) == npatches, (
        f"Number of beta_j0_env values ({len(params.beta_j0_env)}) does not match number of locations ({npatches})"
    )
    # beta_j0_env must be >= 0
    assert np.all(params.beta_j0_env >= 0.0), "beta_j0_env values must be >= 0"

    # length of theta_j must be equal to number of locations
    assert len(params.theta_j) == npatches, (
        f"Number of theta_j values ({len(params.theta_j)}) does not match number of locations ({npatches})"
    )
    # theta_j must be between 0 (no WASH intervention) and 1 (full WASH protection)
    assert np.all((params.theta_j >= 0.0) & (params.theta_j <= 1.0)), "theta_j values must be in the range [0, 1]"

    # shape of psi_jt = (nticks, npatches)
    assert params.psi_jt.shape == (nticks, npatches), (
        f"Shape of psi_jt ({params.psi_jt.shape}) does not match (nticks, npatches) = ({nticks}, {npatches})"
    )

    # psi_jt
    # TODO - TBD

    # zeta_1 and zeta_2 must be >= 0
    assert params.zeta_1 >= 0.0, "zeta_1 value must be >= 0"
    assert params.zeta_2 >= 0.0, "zeta_2 value must be >= 0"
    # TODO - TBD any other limits

    # kappa must be >= 0
    assert params.kappa >= 0.0, "kappa value must be >= 0"

    # delta_min must be >= 0 and delta_max must be >= delta_min
    assert params.delta_min >= 0.0, "delta_min value must be >= 0"
    assert params.delta_max >= params.delta_min, "delta_max value must be >= delta_min"

    return


class Parameters:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        return

    def check(self):
        # assert hasattr(self.model, "patches"), "Parameters: model needs to have a 'patches' attribute."
        # assert hasattr(self.model, "agents"), "Parameters: model needs to have a 'agents' attribute."
        assert hasattr(self.model, "params"), "Parameters: model needs to have a 'params' attribute."

        return

    def __call__(self, _model, _tick):
        pass

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        # Stacked bar chart of initial populations
        # Stacked bar chart of initial populations
        categories = ["S_j_initial", "E_j_initial", "I_j_initial", "R_j_initial", "V1_j_initial", "V2_j_initial"]
        data = [getattr(self.model.params, category) for category in categories]

        x = np.arange(len(self.model.params.location_name))
        bottom = np.zeros(len(self.model.params.location_name))

        for category, values in zip(categories, data):
            plt.bar(x, values, bottom=bottom, label=category)
            bottom += values

        plt.xticks(x, self.model.params.location_name, rotation=45, ha="right")
        plt.xlabel("Location Name")
        plt.ylabel("Population")
        plt.title("Initial Populations by Category")
        plt.legend()
        plt.tight_layout()

        yield

        # Birth rates by location over time
        plt.figure(figsize=(12, 6))
        plt.imshow(self.model.params.b_jt.T, aspect="auto", cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Birth Rate")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.title("Birth Rates by Location Over Time")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.tight_layout()
        # plt.show()
        yield

        # Mortality rates by location over time
        plt.figure(figsize=(12, 6))
        plt.imshow(self.model.params.d_jt.T, aspect="auto", cmap="Reds", interpolation="nearest")
        plt.colorbar(label="Mortality Rate")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.title("Non-Disease Mortality Rates by Location Over Time")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.tight_layout()
        # plt.show()
        yield

        # Vaccination (first dose) rates by location over time
        plt.figure(figsize=(12, 6))
        plt.imshow(self.model.params.nu_1_jt.T, aspect="auto", cmap="Greens", interpolation="nearest")
        plt.colorbar(label="Vaccination Count")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.title("First Dose Vaccination Counts by Location Over Time")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.tight_layout()
        # plt.show()
        yield

        # Vaccination (second dose) rates by location over time
        plt.figure(figsize=(12, 6))
        plt.imshow(self.model.params.nu_2_jt.T, aspect="auto", cmap="Greens", interpolation="nearest")
        plt.colorbar(label="Vaccination Count")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.title("Second Dose Vaccination Counts by Location Over Time")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.tight_layout()
        # plt.show()
        yield

        # Disease mortality rate over time
        plt.figure(figsize=(12, 6))
        plt.imshow(self.model.params.mu_jt.T, aspect="auto", cmap="Reds", interpolation="nearest")
        plt.colorbar(label="Disease Mortality Rate")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.title("Disease Mortality Rate by Location Over Time")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.tight_layout()
        # plt.show()
        yield

        # Seasonality factor by location over time
        plt.figure(figsize=(12, 6))
        plt.imshow(self.model.params.beta_j_seasonality.T, aspect="auto", cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Seasonal Factor")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.title("Seasonal Human-Human Transmission Factor by Location Over Time")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.tight_layout()
        # plt.show()
        yield

        # Emmigration probability rates by location
        plt.figure(figsize=(12, 6))
        plt.scatter(self.model.params.location_name, self.model.params.tau_i, marker="x", color="purple")
        plt.xlabel("Location Name")
        plt.ylabel("Emigration Probability")
        plt.title("Emigration Probabilities by Location")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        yield

        # Spatial connectivity matrix
        plt.figure(figsize=(12, 9))
        plt.imshow(self.model.params.pi_ij, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Connectivity")
        plt.xlabel("Destination Location Index")
        plt.xticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Source Location Index")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.yticks(rotation=45, ha="right")
        plt.title("Spatial Connectivity Matrix (pi_ij)")
        plt.tight_layout()
        yield

        # WASH fraction by location
        plt.figure(figsize=(12, 6))
        plt.scatter(self.model.params.location_name, self.model.params.theta_j, marker="x", color="purple")
        plt.xlabel("Location Name")
        plt.ylabel("WASH Coverage")
        plt.title("WASH Coverage by Location")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        yield

        # Environmental suitability factor by location over time
        plt.figure(figsize=(12, 6))
        plt.imshow(self.model.params.psi_jt.T, aspect="auto", cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Environmental Suitability Factor")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.title("Environmental Suitability Factor by Location Over Time")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.tight_layout()
        # plt.show()
        yield

        return


if __name__ == "__main__":
    # Parameters.test()

    from laser_cholera.metapop.model import Model

    parameters = get_parameters(overrides={"verbose": True})
    model = Model(parameters)
    model.components = [Parameters]
    model.run()
    model.visualize(pdf=False)
