from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from laser_cholera import iso_codes

__CWD__ = Path.cwd()
__DATA_DIR__ = Path(__file__).parent.absolute() / "data"


def write_default_yaml(filename: Path = __CWD__ / "defaults.yaml") -> None:
    """
    Write the default parameters to a YAML file.

    Args:

        filename (Path): The path to the YAML file to write the default parameters to.
    """
    filename = Path(filename)   # allow for string input

    with filename.open("w") as file:
        yaml.dump(
            {
                ## Meta Parameters
                "nticks": 365,
                "verbose": False,
                "outdir": ".",
                "seed": 20241217,

                ## Scalar Parameters
                "alpha": 0.95,  # population mixing parameter
                "delta_min": 1.0 / 3.0,  # minimum environmental decay rate
                "delta_max": 1.0 / 90.0,  # maximum environmental decay rate
                "epsilon": get_default_epsilon(),  # acquired immunity decay rate
                "gamma": 1.0 / 3.0,  # infected recovery rate
                "kappa": 316_000,  # environmental concentration of V. cholerae that results in a 50% probability of infection
                "mu": 0.19,  # - mortality rate for infected individuals due to V. cholerae
                "omega": get_default_omega(),  # rate of waning immunity for vaccinated individuals
                "phi": get_default_phi(),  # effectiveness of OCV
                "sigma": get_default_sigma(),  # fraction of infections that are symptomatic
                "zeta": 200_000,  # rate of shedding from individuals into the environment

                ## Per Patch Parameters
                "birthrate": get_patch_birthrates(),
                "initpop": get_patch_initial_population(),
                "mortrate": get_patch_mortality_rates(),
                "tau": get_patch_taus(),  # probability of emmigration

                ## Time-varying Per Patch Parameters
                "beta_env_t0": get_beta_env_t0(),  # environmental transmission rate
                "beta_hum_t0": get_beta_hum_t0(),  # human-to-human transmission rate

                # "delta": get_patch_deltas(),  # environmental decay rate
                "nu": get_patch_nus(),  # vaccination rate
                "psi": get_patch_psis(),  # environmental transmission rate
                "theta": get_patch_thetas(),  # fraction of population with WASH

                ## Spatial Parameters
                "pi": get_pi_matrix(),  # probability of migration from patch i to patch j

                ## Seasonal Parameters
                "seasonality_coefficients": get_seasonality_coefficients(),  # seasonal factor
            },
            file,
        )

    return filename


def get_default_epsilon(filename=__DATA_DIR__ / "param_epsilon_immune_decay.csv") -> np.float64:
    df = pd.read_csv(filename)
    epsilon = df[(df.parameter_distribution == "point") & (df.parameter_name == "mean")].parameter_value.iloc[0]

    return float(epsilon)


def get_default_omega(filename=__DATA_DIR__ / "param_omega_vaccine_effectiveness.csv") -> np.float64:
    df = pd.read_csv(filename)
    omega = df[(df.parameter_distribution == "point") & (df.parameter_name == "mean")].parameter_value.iloc[0]

    return float(omega)


def get_default_phi(filename=__DATA_DIR__ / "param_phi_vaccine_effectiveness.csv") -> np.float64:
    df = pd.read_csv(filename)
    phi = df[(df.parameter_distribution == "point") & (df.parameter_name == "mean")].parameter_value.iloc[0]

    return float(phi)


def get_default_sigma(filename=__DATA_DIR__ / "param_sigma_prop_symptomatic.csv") -> np.float64:
    df = pd.read_csv(filename)
    alpha = df[(df.parameter_distribution == "beta") & (df.parameter_name == "shape1")].parameter_value.iloc[0]
    beta = df[(df.parameter_distribution == "beta") & (df.parameter_name == "shape2")].parameter_value.iloc[0]
    sigma = float(alpha / (alpha + beta))   # expected value of a beta distribution = alpha / (alpha + beta)

    return sigma


def get_patch_birthrates(filename=__DATA_DIR__ / "demographics_africa_2000_2023.csv") -> dict:
    demographics = pd.read_csv(filename)
    latest = demographics[demographics.year == demographics.year.max()]
    rates = {iso: rate for iso, rate in zip(latest.iso_code, latest.birth_rate_per_day) if iso in iso_codes}

    return rates


def get_patch_initial_population(filename=__DATA_DIR__ / "demographics_africa_2000_2023.csv") -> dict:
    demographics = pd.read_csv(filename)
    latest = demographics[demographics.year == demographics.year.max()]
    populations = {iso: pop for iso, pop in zip(latest.iso_code, latest.population) if iso in iso_codes}

    return populations


def get_patch_mortality_rates(filename=__DATA_DIR__ / "demographics_africa_2000_2023.csv") -> dict:
    demographics = pd.read_csv(filename)
    latest = demographics[demographics.year == demographics.year.max()]
    rates = {iso: rate for iso, rate in zip(latest.iso_code, latest.death_rate_per_day) if iso in iso_codes}

    return rates


def get_patch_taus(filename=__DATA_DIR__ / "param_tau_departure.csv") -> dict:
    df = pd.read_csv(filename)
    point_values = df[df.parameter_distribution == "point"]
    taus = {iso: tau for iso, tau in zip(point_values.i, point_values.parameter_value) if iso in iso_codes}

    return taus


########## TODO ##########


def get_beta_env_t0() -> dict:
    betas = {iso: 1.0 for iso in iso_codes}

    return betas


def get_beta_hum_t0() -> dict:
    betas = {iso: 1.0 for iso in iso_codes}

    return betas


# def get_patch_deltas(filename=__DATA_DIR__ / "param_delta_env_decay.csv") -> dict:
#     df = pd.read_csv(filename)
#     point_values = df[df.parameter_distribution == "point"]
#     deltas = {iso: delta for iso, delta in zip(point_values.iso_code, point_values.delta) if iso in iso_codes}

#     return deltas


# **** TODO ****
def get_patch_nus(filename=__DATA_DIR__ / "param_nu_vaccination_rate.csv") -> dict:
    df = pd.read_csv(filename)
    point_values = df[df.parameter_distribution == "point"]
    nus = {iso: nu for iso, nu in zip(point_values.j, point_values.parameter_value) if iso in iso_codes}

    return nus


def get_patch_psis(filename=__DATA_DIR__ / "pred_psi_suitability.csv") -> dict:
    df = pd.read_csv(filename)
    ssa = df[[iso in iso_codes for iso in df.iso_code]]
    # consider using "pred_smooth" instead of "pred"
    pivot = ssa.pivot(index="iso_code", columns="date_start", values="pred")
    psis = {iso: list(pivot.loc[iso]) for iso in iso_codes}

    return psis


def get_patch_thetas(filename=__DATA_DIR__ / "param_theta_wash.csv") -> dict:
    df = pd.read_csv(filename)
    point_values = df[df.parameter_distribution == "point"]
    point_values.rename(columns={"j": "iso_code"}, inplace=True)
    thetas = {iso: theta for iso, theta in zip(point_values.iso_code, point_values.parameter_value) if iso in iso_codes}

    return thetas


def get_pi_matrix(filename=__DATA_DIR__ / "param_pi_diffusion.csv") -> np.ndarray:
    df = pd.read_csv(filename)
    pivot = df.pivot(index="i", columns="j", values="parameter_value")
    # zero out self-migration
    # use `iso_codes` to ensure all countries are accounted for
    for iso in iso_codes:
        pivot.loc[iso, iso] = 0.0
    π = {iso: list(pivot.loc[iso][iso_codes]) for iso in iso_codes}  # noqa: PLC2401

    return π


def get_seasonality_coefficients(filename=__DATA_DIR__ / "param_seasonal_dynamics.csv") -> dict:
    df = pd.read_csv(filename)
    precipitation = df[df.response == "precipitation"]
    pivot = precipitation.pivot(index="country_iso_code", columns="parameter", values="mean")
    # Note: coefficients are in a1, a2, b1, b2 order
    coefficients = {iso: list(pivot.loc[iso]) for iso in iso_codes}

    return coefficients


if __name__ == "__main__":
    print(f"Wrote default YAML to `{write_default_yaml()}`")
