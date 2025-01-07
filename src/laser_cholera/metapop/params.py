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

    # IDs are 1-based not 0-based like indices
    # Map string IDs to names and names to indices (0-based)

    id_to_location_map = {str(id): name for id, name in zip(parameters["location_id"], parameters["location_name"])}
    location_to_idx_map = {name: idx for idx, name in enumerate(sorted(parameters["location_name"]))}
    id_to_idx_map = {str(id): location_to_idx_map[name] for id, name in id_to_location_map.items()}

    params = PropertySet(parameters)

    params.date_start = datetime.strptime(params.date_start, "%Y-%m-%d")  # noqa: DTZ007
    params.date_stop = datetime.strptime(params.date_stop, "%Y-%m-%d")  # noqa: DTZ007

    params.location_id = np.array(params.location_id, dtype=np.uint32)

    params.N_j_initial = np.array(params.N_j_initial, dtype=np.uint32)
    params.S_j_initial = np.array(params.S_j_initial, dtype=np.uint32)
    params.I_j_initial = np.array(params.I_j_initial, dtype=np.uint32)
    params.R_j_initial = np.array(params.R_j_initial, dtype=np.uint32)

    params.b_j = np.array(params.b_j, dtype=np.float32)
    params.d_j = np.array(params.d_j, dtype=np.float32)

    width = max(map(len, params.nu_jt.values()))
    height = len(params.nu_jt)
    nu_jt = np.zeros((height, width), dtype=np.float32)
    for key, values in params.nu_jt.items():
        nu_jt[id_to_idx_map[key], : len(values)] = values
    params.nu_jt = nu_jt

    params.beta_j0_hum = np.array(params.beta_j0_hum, dtype=np.float32)

    width = max(map(len, params.beta_j_seasonality.values()))
    height = len(params.beta_j_seasonality)
    beta_j_seasonality = np.zeros((height, width), dtype=np.float32)
    for key, values in params.beta_j_seasonality.items():
        beta_j_seasonality[id_to_idx_map[key], : len(values)] = values
    params.beta_j_seasonality = beta_j_seasonality

    params.tau_i = np.array(params.tau_i, dtype=np.float32)

    width = max(map(len, params.pi_ij.values()))
    height = len(params.pi_ij)
    pi_ij = np.zeros((height, width), dtype=np.float32)
    for key, values in params.pi_ij.items():
        pi_ij[id_to_idx_map[key], : len(values)] = [value if value != "NA" else 0.0 for value in values]
    params.pi_ij = pi_ij

    params.beta_j0_env = np.array(params.beta_j0_env, dtype=np.float32)
    params.theta_j = np.array(params.theta_j, dtype=np.float32)

    width = max(map(len, params.psi_jt.values()))
    height = len(params.psi_jt)
    psi_jt = np.zeros((height, width), dtype=np.float32)
    for key, values in params.psi_jt.items():
        psi_jt[id_to_idx_map[key], : len(values)] = values
    params.psi_jt = psi_jt

    if overrides is not None:
        params += overrides

        if params.verbose:
            click.echo(f"Updated/overrode file parameters with `{overrides}`…")

    if params.verbose:
        click.echo(f"Loaded parameters from `{filename}`…")

    return params
