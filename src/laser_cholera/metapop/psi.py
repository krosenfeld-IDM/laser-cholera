"""Environmental transmission rate."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


class Psi:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Psi: model needs to have a 'agents' attribute."
        assert hasattr(model.agents, "S"), "Psi: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(model, "patches"), "Psi: model needs to have a 'patches' attribute."
        assert hasattr(model.patches, "PSI"), "Psi: model patches needs to have a 'PSI' (environmental transmission rate) attribute."
        assert hasattr(
            model.patches, "beta_env"
        ), "Psi: model patches needs to have a 'beta_env' (environmental transmission rate) attribute."
        assert hasattr(model.patches, "W"), "Psi: model patches needs to have a 'W' (environmental) attribute."
        assert hasattr(model.patches, "tau"), "Psi: model patches needs to have a 'tau' (contamination rate) attribute."
        assert hasattr(model.patches, "theta"), "Psi: model patches needs to have a 'theta' (fraction of population with WASH) attribute."
        assert hasattr(model, "params"), "Psi: model needs to have a 'params' attribute."
        assert hasattr(model.params, "kappa"), "Psi: model params needs to have a 'kappa' (environmental transmission rate) parameter."

        initialize_psi(model)

        return

    def __call__(self, model, tick: int) -> None:
        PSIprime = model.patches.PSI[tick + 1]
        beta_env = model.patches.beta_env[tick]
        S = model.agents.S[tick]
        W = model.patches.W[tick]

        PSIprime[:] = beta_env * (S * (1 - model.patches.tau)) * (1 - model.patches.theta[tick]) * W
        PSIprime /= model.params.kappa + W

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Environmental Transmission Rate in Patch {ipatch}")
        plt.plot(self.model.patches.PSI[:, ipatch], color="orange", label="Environmental Transmission Rate")
        plt.xlabel("Tick")

        yield
        return range(0)


def initialize_psi(model):
    csvfile = Path(__file__).parent / "data" / "pred_psi_suitability.csv"
    df = pd.read_csv(csvfile)[["iso_code", "date_start", "pred"]]
    df.date_start = pd.to_datetime(df.date_start)
    pivot = df.pivot(index="iso_code", columns="date_start", values="pred")
    countries = model.scenario[["ISO"]]
    data = countries.merge(pivot, left_on="ISO", right_on="iso_code")
    data = data[data.columns[1:]]  # drop the ISO code column
    model.patches.PSI[:, :] = data[data.columns[1:]].to_numpy()[:, -(model.params.nticks + 1) :].T

    return
