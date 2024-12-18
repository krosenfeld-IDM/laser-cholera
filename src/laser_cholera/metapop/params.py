import re
from pathlib import Path
from typing import Optional
from typing import Union

import click
import yaml
from laser_core.propertyset import PropertySet

from .defaults import write_default_yaml


def get_parameters(filename: Optional[Union[str, Path]] = None, overrides:Optional[str] = None) -> PropertySet:

    filename = Path(filename) if filename is not None else write_default_yaml()

    with filename.open("r") as file:
        parameters = yaml.safe_load(file)

    params = PropertySet(parameters)
    params.birthrate = PropertySet(params.birthrate)
    params.initpop = PropertySet(params.initpop)
    params.mortrate = PropertySet(params.mortrate)
    params.tau = PropertySet(params.tau)
    params.beta_env_t0 = PropertySet(params.beta_env_t0)
    params.beta_hum_t0 = PropertySet(params.beta_hum_t0)
    params.nu = PropertySet(params.nu)
    params.psi = PropertySet(params.psi)
    params.theta = PropertySet(params.theta)
    params.pi = PropertySet(params.pi)
    params.seasonality_coefficients = PropertySet(params.seasonality_coefficients)

    if params.verbose:
        click.echo(f"Loaded parameters from `{filename}`…")

    # Overwrite any parameters with those from the command line (optional)
    for kvp in overrides:
        key, value = re.split("[=:]+", kvp)
        if key not in params:
            click.echo(f"Unknown parameter `{key}` ({value=}). Skipping…")
            continue
        value = type(params[key])(value)    # Cast the value to the same type as the existing parameter
        click.echo(f"Using `{value}` for parameter `{key}` from the command line…")
        params[key] = value

    return params
