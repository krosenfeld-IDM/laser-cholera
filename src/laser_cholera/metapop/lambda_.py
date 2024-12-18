"""Human-to-human transmission rate."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class Lambda_:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Lambda_: model needs to have a 'agents' attribute."
        assert hasattr(model.agents, "S"), "Lambda_: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(model.agents, "I"), "Lambda_: model agents needs to have a 'I' (infectious) attribute."
        assert hasattr(model.agents, "N"), "Lambda_: model agents needs to have a 'N' (current agents) attribute."
        assert hasattr(model, "patches"), "Lambda_: model needs to have a 'patches' attribute."
        assert hasattr(model.patches, "network"), "Lambda_: model patches needs to have a 'network' attribute."
        assert hasattr(model.patches, "beta_hum"), "Lambda_: model patches needs to have a 'beta_hum' (human-to-human rate) attribute."
        assert hasattr(model, "params"), "Lambda_: model needs to have a 'params' attribute."
        assert "alpha" in model.params, "Lambda_: model params needs to have an 'alpha' parameter."

        model.patches.add_vector_property("LAMBDA", length=model.params.nticks + 1, dtype=float, default=0.0)
        model.patches.add_scalar_property("tau", dtype=float, default=0.0)
        initialize_tau(model, model.params)
        model.patches.add_vector_property("pi", length=model.patches.count, dtype=float, default=0.0)
        model.patches.pi[:, :] = model.patches.network

        return

    def __call__(self, model, tick: int) -> None:
        r"""
        Calculate the current human-to-human transmission rate per patch.

        $\Lambda_{j,t+1} = \frac {\beta^{hum}_{jt}((S_{jt}(1 - \tau_j))(I_{jt}(1 - \tau_j) + \sum_{\forall i \neq j (\pi_{ij} \tau_j I_{it})}))^{\alpha}} {N_{jt}}$
        """

        # "lambda" is a reserved keyword in Python
        LAMBDAprime = model.patches.LAMBDA[tick + 1]
        LAMBDAprime[:] = model.agents.I[tick] * (1 - model.patches.tau)
        # TODO - sum over axis=0 or axis=1?
        LAMBDAprime += (model.patches.pi * (model.patches.tau * model.agents.I[tick])).sum(axis=0)
        LAMBDAprime *= model.agents.S[tick] * (1 - model.patches.tau)
        np.power(LAMBDAprime, model.params.alpha, out=LAMBDAprime)
        LAMBDAprime *= model.patches.beta_hum[tick]
        LAMBDAprime /= model.agents.N[tick]

        return

    def plot(self, fig: Figure = None):
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Patch {ipatch} - Human-to-Human Transmission Rate")
        plt.plot(self.model.patches.LAMBDA[:, ipatch], color="blue", label="Human-to-Human Transmission Rate")
        plt.xlabel("Tick")

        yield
        return


def initialize_tau(model, parameters):
    csvfile = Path(__file__).parent / "data" / "param_tau_departure.csv"
    df = pd.read_csv(csvfile)
    point_values = df[df.parameter_distribution == "point"][["i", "parameter_value"]]
    merged = pd.merge(model.scenario[["ISO"]], point_values, left_on="ISO", right_on="i")
    model.patches.tau[:] = merged.parameter_value

    return
