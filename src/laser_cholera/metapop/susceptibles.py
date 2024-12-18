import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Susceptibles:
    def __init__(self, model, verbose: bool = False):
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Susceptibles: model needs to have a 'agents' attribute."
        assert hasattr(model.agents, "S"), "Susceptibles: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(model.agents, "R"), "Susceptibles: model agents needs to have a 'R' (recovered) attribute."
        assert hasattr(model.agents, "N"), "Susceptibles: model agents needs to have a 'N' (current population) attribute."
        assert hasattr(model.agents, "V"), "Susceptibles: model agents needs to have a 'V' (vaccinated) attribute."
        assert hasattr(model, "patches"), "Susceptibles: model needs to have a 'patches' attribute."
        assert hasattr(
            model.patches, "LAMBDA"
        ), "Susceptibles: model patches needs to have a 'LAMBDA' (human-to-human transmission rate) attribute."
        assert hasattr(
            model.patches, "PSI"
        ), "Susceptibles: model patches needs to have a 'PSI' (environmental transmission rate) attribute."
        assert hasattr(model.patches, "nu"), "Susceptibles: model patches needs to have a 'nu' (incidence) attribute."
        assert hasattr(model.patches, "birthrate"), "Susceptibles: model patches needs to have a 'birth' parameter."
        assert hasattr(model.patches, "mortrate"), "Susceptibles: model patches needs to have a 'mortrate' parameter."
        assert hasattr(model, "params"), "Infected: model needs to have a 'params' attribute."
        assert hasattr(model.params, "phi"), "Susceptibles: model params needs to have a 'phi' (death rate) parameter."
        assert hasattr(model.params, "omega"), "Susceptibles: model params needs to have a 'omega' (vaccination rate) parameter."

        # model.agents.add_vector_property("S", length=model.params.nticks+1, dtype=float, default=0.0)
        model.agents.S[0,:] = model.patches.initpop

        return

    def __call__(self, model, tick: int) -> None:
        S = model.agents.S[tick]
        Sprime = model.agents.S[tick + 1]
        # R = model.agents.R[tick]
        N = model.agents.N[tick]
        # V = model.agents.V[tick]
        # LAMBDA = model.patches.LAMBDA[tick + 1]
        # PSI = model.patches.PSI[tick + 1]

        Sprime[:] = S
        # births
        Sprime += model.prng.binomial(N, model.patches.birthrate).astype(Sprime.dtype)  # rate or probability?
        # # vaccinations
        # Sprime -= model.prng.binomial(S, model.params.phi * model.patches.nu[tick]).astype(Sprime.dtype)
        # # waning vaccinations
        # Sprime += model.prng.binomial(V, model.params.omega).astype(Sprime.dtype)
        # human-to-human transmission
        # Sprime -= LAMBDA.astype(Sprime.dtype)  # TODO - is this a rate or a count?
        # # environmental-to-human transmission
        # Sprime -= PSI.astype(Sprime.dtype)  # TODO - is this a rate or a count?
        # # waning acquired immunity
        # Sprime += model.prng.binomial(R, model.params.epsilon).astype(Sprime.dtype)  # rate or probability?
        # non-disease mortality
        Sprime -= model.prng.binomial(S, model.patches.mortrate).astype(Sprime.dtype)  # rate or probability?

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Susceptibles in Patch {ipatch}")
        plt.plot(self.model.agents.S[:, ipatch], color="green", label="Susceptibles")
        plt.xlabel("Tick")

        yield
        return
