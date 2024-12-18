import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Infected:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Infected: model needs to have a 'agents' attribute."
        assert hasattr(model.agents, "I"), "Infected: model agents needs to have a 'I' (infectious) attribute."
        assert hasattr(model, "patches"), "Infected: model needs to have a 'patches' attribute."
        assert hasattr(
            model.patches, "LAMBDA"
        ), "Infected: model patches needs to have a 'LAMBDA' (human-to-human transmission rate) attribute."
        assert hasattr(model.patches, "PSI"), "Infected: model patches needs to have a 'PSI' (environmental transmission rate) attribute."
        assert hasattr(model.patches, "mortrate"), "Infected: model patches needs to have a 'mortrate' parameter."
        assert hasattr(model, "params"), "Infected: model needs to have a 'params' attribute."
        assert hasattr(model.params, "gamma"), "Infected: model params needs to have a 'gamma' (recovery rate) parameter."
        assert hasattr(model.params, "mu"), "Infected: model params needs to have a 'mu' (environmental transmission rate) parameter."
        assert hasattr(
            model.params, "sigma"
        ), "Infected: model params needs to have a 'sigma' (human-to-human transmission rate) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        I = model.agents.I[tick]  # noqa: E741
        Iprime = model.agents.I[tick + 1]
        LAMBDA = model.patches.LAMBDA[tick + 1]
        PSI = model.patches.PSI[tick + 1]
        Sprime = model.agents.S[tick + 1]

        Iprime[:] = I
        # human-to-human transmission
        new_infs = LAMBDA.astype(Iprime.dtype)  # TODO - is this a rate or a count?
        # environmental-to-human transmission
        new_infs += PSI.astype(Iprime.dtype)  # TODO - is this a rate or a count?
        Iprime += new_infs
        Sprime -= new_infs
        # recovered
        Iprime -= model.prng.binomial(I, model.params.gamma).astype(Iprime.dtype)
        # disease deaths
        Iprime -= model.prng.binomial(I, model.params.mu * model.params.sigma).astype(Iprime.dtype)
        # non-disease mortality
        Iprime -= model.prng.binomial(I, model.patches.mortrate).astype(Iprime.dtype)

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig
        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Infected in Patch {ipatch}")
        plt.plot(self.model.agents.I[:, ipatch], color="red", label="Infected")
        plt.xlabel("Tick")

        yield
        return
