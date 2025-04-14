import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Analyzer:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        return

    def check(self):
        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="SIRV Trajectories (Largest Patch)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-1:]:
            for channel in ["S", "Isym", "Iasym", "R", "V1", "V2"]:
                plt.plot(getattr(self.model.agents, channel)[:, ipatch], label=f"{channel}")
        plt.xlabel("Tick")
        plt.ylabel("Population")
        plt.legend()

        yield
        return
