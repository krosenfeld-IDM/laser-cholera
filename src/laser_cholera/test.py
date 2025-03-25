from matplotlib.figure import Figure


class Eradication:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        return

    def check(self):
        return

    def __call__(self, model, tick: int) -> None:
        if tick == 1:
            model.agents.S[tick] += model.agents.Isym[tick] + model.agents.Iasym[tick]
            model.agents.Isym[tick] = 0
            model.agents.Iasym[tick] = 0

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        yield
        return
