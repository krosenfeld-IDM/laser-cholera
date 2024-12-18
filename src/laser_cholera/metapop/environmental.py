import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Environmental:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Environmental: model needs to have a 'agents' attribute."
        assert hasattr(model.agents, "I"), "Environmental: model agents needs to have a 'I' (infectious) attribute."
        assert hasattr(model, "params"), "Environmental: model needs to have a 'params' attribute."
        assert hasattr(model.params, "delta"), "Environmental: model paraams needs to have a 'delta' (suitability decay) parameter."
        assert hasattr(
            model.params, "zeta"
        ), "Environmental: model params needs to have a 'zeta' (environmental transmission rate) parameter."
        assert hasattr(model, "patches"), "Environmental: model needs to have a 'patches' attribute."
        assert hasattr(model.patches, "W"), "Environmental: model patches needs to have a 'W' (environmental) attribute."

        return

    def __call__(self, model, tick: int) -> None:
        W = model.patches.W[tick]
        Wprime = model.patches.W[tick + 1]
        I = model.agents.I[tick]  # noqa: E741

        Wprime[:] = W
        Wprime += model.params.zeta * I
        Wprime -= model.params.delta * W

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Environmental in Patch {ipatch}")
        plt.plot(self.model.patches.W[:, ipatch], color="orange", label="Environmental")
        plt.xlabel("Tick")

        yield
        return
