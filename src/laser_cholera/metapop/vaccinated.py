import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Vaccinated:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Vaccinated: model needs to have a 'agents' attribute."
        assert hasattr(model.agents, "V"), "Vaccinated: model agents needs to have a 'V' (vaccinated) attribute."
        assert hasattr(model.agents, "S"), "Vaccinated: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(model, "params"), "Vaccinated: model needs to have a 'params' attribute."
        assert hasattr(model.params, "phi"), "Vaccinated: model needs to have a 'phi' (vaccination rate) parameter."
        assert hasattr(model.params, "omega"), "Vaccinated: model needs to have a 'omega' (vaccination rate) parameter."
        assert hasattr(model, "patches"), "Vaccinated: model needs to have a 'patches' attribute."
        assert hasattr(model.patches, "mortrate"), "Vaccinated: model patches needs to have a 'mortrate' parameter."
        assert hasattr(model.patches, "nu"), "Vaccinated: model patches needs to have a 'nu' (incidence) attribute."

        return

    def __call__(self, model, tick: int) -> None:
        V = model.agents.V[tick]
        Vprime = model.agents.V[tick + 1]
        S = model.agents.S[tick]
        Sprime = model.agents.S[tick + 1]

        Vprime[:] = V
        newly_vaxxed = model.prng.binomial(S, model.params.phi * model.patches.nu[tick]).astype(Vprime.dtype)  # rate or probability?
        Vprime += newly_vaxxed
        Sprime -= newly_vaxxed
        waned_vax = model.prng.binomial(V, model.params.omega).astype(Vprime.dtype)
        Vprime -= waned_vax
        Sprime += waned_vax
        # non-disease mortality
        Vprime -= model.prng.binomial(V, model.patches.mortrate).astype(Vprime.dtype)  # rate or probability?

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Vaccinated in Patch {ipatch}")
        plt.plot(self.model.agents.V[:, ipatch], color="blue", label="Vaccinated")
        plt.xlabel("Tick")

        yield
        return
