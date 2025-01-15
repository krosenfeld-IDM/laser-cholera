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

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("SIRV Trajectories")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-1:]:
            for channel in ["S", "I", "R", "V"]:
                plt.plot(getattr(self.model.agents, channel)[:, ipatch], label=f"{channel}")
            # plt.plot(self.model.agents.V[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return
