from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class Wash:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "patches"), "Wash: model needs to have a 'population' attribute."

        model.patches.add_vector_property("theta", length=model.params.nticks, dtype=np.float32, default=0.0)

        initialize_theta(model, model.params)

        return

    # def __call__(self, model, tick: int) -> None:
    #     return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig
        plt.title("WASH in Patches")
        plt.plot(self.model.patches.theta, color="green", label="WASH")
        plt.xlabel("Patch")

        yield
        return


def initialize_theta(model, parameters):
    csvfile = Path(__file__).parent / "data" / "param_theta_WASH.csv"
    df = pd.read_csv(csvfile)
    only_points = df[df.parameter_distribution == "point"]
    unique = only_points.drop_duplicates(subset=["j"])
    minimal = unique[["j", "parameter_value"]]
    all_patches = pd.concat(
        [minimal, pd.DataFrame([[iso, 0.0] for iso in (set(model.scenario.ISO) - set(minimal.j))], columns=["j", "parameter_value"])]
    )
    merged = pd.merge(model.scenario[["ISO"]], all_patches, left_on="ISO", right_on="j")
    # Current data has no temporal dimension, so broadcast the parameter value to all ticks
    model.patches.theta[:, :] = np.broadcast_to(merged.parameter_value, model.patches.theta.shape)

    return
