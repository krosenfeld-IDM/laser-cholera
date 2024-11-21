import click
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class PropagatePopulation:
    def __init__(self, model, verbose: bool = False):
        if verbose:
            click.echo("Initializing PropagatePopulation component...")

        assert hasattr(model.patches, "population"), "PropagatePopulation: model.patches needs to have a 'population' attribute."
        assert "nticks" in model.params, "PropagatePopulation: model.params needs to have a 'nticks' attribute."

        self.model = model
        self.verbose = verbose

        return

    def __call__(self, model, tick: int) -> None:
        if tick + 1 < model.params.nticks:
            model.patches.population[tick + 1, :] = model.patches.population[tick, :]

        return

    def plot(self, fig: Figure = None):
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        npatches = 3
        ipatches = self.model.patches.population[0, :].argsort()[-npatches:]
        for i in ipatches:
            plt.plot(self.model.patches.population[:, i], label=f"Patch {i}")
        plt.title(f"Population Over Time\n{npatches} largest patches")

        yield
        return
