import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing import Union

import click
import numpy as np
from laser_core.propertyset import PropertySet


def get_parameters(filename: Optional[Union[str, Path]] = None, overrides: Optional[dict] = None) -> PropertySet:
    filename = Path(filename) if filename is not None else Path(__file__).parent / "data" / "default_parameters.json"

    with filename.open("r") as file:
        parameters = json.load(file)

    # Note the following canonicalizes the order of the locations based on the
    # order in the JSON file.
    # We might consider either
    # a) alphabetical order by location name or
    # b) order by int(ID)

    params = PropertySet(parameters)

    num_patches = len(params.location_name)
    assert len(params.location_id) == num_patches, (
        f"Number of location IDs ({len(params.location_id)}) does not match number of location names ({num_patches})"
    )

    # IDs are 1-based not 0-based like indices
    # Map string IDs to names and names to indices (0-based)

    strid_to_location_map = {str(id): name for id, name in zip(parameters["location_id"], parameters["location_name"])}
    location_to_idx_map = {name: idx for idx, name in enumerate(sorted(parameters["location_name"]))}
    strid_to_numidx_map = {id: location_to_idx_map[name] for id, name in strid_to_location_map.items()}

    params.date_start = datetime.strptime(params.date_start, "%Y-%m-%d")  # noqa: DTZ007
    params.date_stop = datetime.strptime(params.date_stop, "%Y-%m-%d")  # noqa: DTZ007
    params.nticks = (params.date_stop - params.date_start).days + 1
    print(f"Simulation calendar dates: {params.date_start} to {params.date_stop} ({params.nticks} ticks)")

    params.location_id = np.array(params.location_id, dtype=np.uint32)
    print(f"{len(params.location_id)=}")
    print(f"{len(params.location_name)=}")

    params.N_j_initial = np.array(params.N_j_initial, dtype=np.uint32)
    params.S_j_initial = np.array(params.S_j_initial, dtype=np.uint32)
    params.E_j_initial = np.array(params.E_j_initial, dtype=np.uint32)
    params.I_j_initial = np.array(params.I_j_initial, dtype=np.uint32)
    params.R_j_initial = np.array(params.R_j_initial, dtype=np.uint32)
    params.V1_j_initial = np.array(params.V1_j_initial, dtype=np.uint32)
    params.V2_j_initial = np.array(params.V2_j_initial, dtype=np.uint32)

    assert len(params.N_j_initial) == num_patches, (
        f"Number of N_j_initial values ({len(params.N_j_initial)}) does not match number of locations ({num_patches})"
    )
    assert len(params.S_j_initial) == num_patches, (
        f"Number of S_j_initial values ({len(params.S_j_initial)}) does not match number of locations ({num_patches})"
    )
    assert len(params.E_j_initial) == num_patches, (
        f"Number of E_j_initial values ({len(params.E_j_initial)}) does not match number of locations ({num_patches})"
    )
    assert len(params.I_j_initial) == num_patches, (
        f"Number of I_j_initial values ({len(params.I_j_initial)}) does not match number of locations ({num_patches})"
    )
    assert len(params.R_j_initial) == num_patches, (
        f"Number of R_j_initial values ({len(params.R_j_initial)}) does not match number of locations ({num_patches})"
    )
    assert len(params.V1_j_initial) == num_patches, (
        f"Number of V1_j_initial values ({len(params.V1_j_initial)}) does not match number of locations ({num_patches})"
    )
    assert len(params.V2_j_initial) == num_patches, (
        f"Number of V2_j_initial values ({len(params.V2_j_initial)}) does not match number of locations ({num_patches})"
    )

    # TODO - sanity check the following
    # assert params.N_j
    # assert params.S_j
    # assert params.E_j
    # assert params.I_j
    # assert params.R_j
    # assert params.V1_j
    # assert params.V2_j

    width = max(map(len, params.b_jt.values()))
    height = len(params.b_jt)
    assert width == params.nticks, f"Number of b_jt values ({width}) does not match number of ticks ({params.nticks})"
    assert height == num_patches, f"Number of b_jt keys ({height}) does not match number of locations ({num_patches})"
    b_jt = np.zeros((height, width), dtype=np.float32)
    for key, values in params.b_jt.items():
        b_jt[strid_to_numidx_map[key], : len(values)] = values
    params.b_jt = b_jt

    # TODO - sanity check values of b_jt
    # assert params.b_jt

    width = max(map(len, params.d_jt.values()))
    height = len(params.d_jt)
    assert width == params.nticks, f"Number of d_jt values ({width}) does not match number of ticks ({params.nticks})"
    assert height == num_patches, f"Number of d_jt keys ({height}) does not match number of locations ({num_patches})"
    d_jt = np.zeros((height, width), dtype=np.float32)
    for key, values in params.d_jt.items():
        d_jt[strid_to_numidx_map[key], : len(values)] = values
    params.d_jt = d_jt

    # TODO - sanity check values of d_jt
    # assert params.d_jt

    cticks = max(map(len, params.nu_1_jt.values()))
    cnodes = len(params.nu_1_jt)
    assert cticks == params.nticks, f"Number of nu_1_jt values ({cticks}) does not match number of ticks ({params.nticks})"
    assert cnodes == num_patches, f"Number of nu_1_jt keys ({cnodes}) does not match number of locations ({num_patches})"
    assert all(len(row) == cticks for row in params.nu_1_jt.values()), (
        f"Some nu_1_jt row value counts do not match number of ticks ({params.nticks})"
    )
    # Transpose here for more efficient access by tick
    nu_1_jt = np.zeros((cticks, cnodes), dtype=np.float32)
    for key, values in params.nu_1_jt.items():
        nu_1_jt[:, strid_to_numidx_map[key]] = values
    params.nu_1_jt = nu_1_jt  # Change from dict to np.ndarray

    # TODO - sanity check values of nu_1_jt
    # assert params.nu_1_jt

    cticks = max(map(len, params.nu_2_jt.values()))
    cnodes = len(params.nu_2_jt)
    assert cticks == params.nticks, f"Number of nu_2_jt values ({cticks}) does not match number of ticks ({params.nticks})"
    assert cnodes == num_patches, f"Number of nu_2_jt keys ({cnodes}) does not match number of locations ({num_patches})"
    assert all(len(row) == cticks for row in params.nu_2_jt.values()), (
        f"Some nu_2_jt row value counts do not match number of ticks ({params.nticks})"
    )
    # Transpose here for more efficient access by tick
    nu_2_jt = np.zeros((cticks, cnodes), dtype=np.float32)
    for key, values in params.nu_2_jt.items():
        nu_2_jt[:, strid_to_numidx_map[key]] = values
    params.nu_2_jt = nu_2_jt  # Change from dict to np.ndarray

    # TODO - sanity check values of nu_2_jt
    # assert params.nu_2_jt

    # TODO - validate the following
    # assert params.phi_1
    # assert params.phi_2
    # assert params.omega_1
    # assert params.omega_2
    # assert params.epsilon
    # assert params.gamma_1
    # assert params.gamma_2
    # assert params.mu
    # assert params.rho
    # assert params.sigma
    # assert params.alpha_1
    # assert params.alpha_2
    # assert params.zeta_1
    # assert params.zeta_2
    # assert params.kappa
    # assert params.delta_min
    # assert params.delta_max

    params.beta_j0_hum = np.array(params.beta_j0_hum, dtype=np.float32)
    assert len(params.beta_j0_hum) == num_patches, (
        f"Number of beta_j0_hum values ({len(params.beta_j0_hum)}) does not match number of locations ({num_patches})"
    )

    # TODO - sanity check values of beta_j0_hum
    # assert params.beta_j0_hum

    width = max(map(len, params.beta_j_seasonality.values()))
    height = len(params.beta_j_seasonality)
    assert width == 366, f"Number of beta_j_seasonality values ({width}) does not match days in year (366)"
    assert height == num_patches, f"Number of beta_j_seasonality keys ({height}) does not match number of locations ({num_patches})"
    beta_j_seasonality = np.zeros((height, width), dtype=np.float32)
    for key, values in params.beta_j_seasonality.items():
        beta_j_seasonality[strid_to_numidx_map[key], : len(values)] = values
    params.beta_j_seasonality = beta_j_seasonality

    # TODO - sanity check values of beta_j_seasonality
    # assert params.beta_j_seasonality

    params.tau_i = np.array(params.tau_i, dtype=np.float32)
    assert len(params.tau_i) == num_patches, (
        f"Number of tau_i values ({len(params.tau_i)}) does not match number of locations ({num_patches})"
    )
    assert np.all((params.tau_i >= 0.0) & (params.tau_i <= 1.0)), "tau_i values must be in the range [0, 1]"

    width = max(map(len, params.pi_ij.values()))
    height = len(params.pi_ij)
    assert width == num_patches, f"Number of pi_ij values ({width}) does not match number of locations ({num_patches})"
    assert height == num_patches, f"Number of pi_ij keys ({height}) does not match number of locations ({num_patches})"
    pi_ij = np.zeros((height, width), dtype=np.float32)
    for key, values in params.pi_ij.items():
        pi_ij[strid_to_numidx_map[key], : len(values)] = [value if value != "NA" else 0.0 for value in values]
    params.pi_ij = pi_ij

    assert np.all((params.pi_ij >= 0.0) & (params.pi_ij <= 1.0)), "pi_ij values must be in the range [0, 1]"
    assert np.all(np.diag(params.pi_ij) == 0.0), "pi_ij diagonal values must be 0.0"
    assert np.all(np.sum(params.pi_ij, axis=1) <= 1.0), "pi_ij row sums must be 1.0"

    # TODO - validate the following
    # assert params.pi

    params.beta_j0_env = np.array(params.beta_j0_env, dtype=np.float32).reshape(-1, 1)
    params.theta_j = np.array(params.theta_j, dtype=np.float32)

    assert len(params.beta_j0_env) == num_patches, (
        f"Number of beta_j0_env values ({len(params.beta_j0_env)}) does not match number of locations ({num_patches})"
    )
    assert len(params.theta_j) == num_patches, (
        f"Number of theta_j values ({len(params.theta_j)}) does not match number of locations ({num_patches})"
    )

    # TODO - sanity check values of beta_j0_env and theta_j
    # assert params.beta_j0_env
    # assert params.theta_j

    width = max(map(len, params.psi_jt.values()))
    height = len(params.psi_jt)
    assert width == params.nticks, f"Number of psi_jt values ({width}) does not match number of ticks ({params.nticks})"
    assert height == num_patches, f"Number of psi_jt keys ({height}) does not match number of locations ({num_patches})"
    psi_jt = np.zeros((height, width), dtype=np.float32)
    for key, values in params.psi_jt.items():
        psi_jt[strid_to_numidx_map[key], : len(values)] = values
    params.psi_jt = psi_jt

    # TODO - sanity check values of psi_jt
    # assert params.psi_jt

    if overrides is not None:
        params += overrides

        if params.verbose:
            click.echo(f"Updated/overrode file parameters with `{overrides}`…")

    if params.verbose:
        click.echo(f"Loaded parameters from `{filename}`…")

    return params
