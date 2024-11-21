import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Births:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        assert (
            getattr(model.patches, "population", None) is not None
        ), "The Births component requires the model patches to have a `population` attribute"
        model.patches.add_scalar_property("daily_birth_rate", dtype=np.float32, default=np.float32(0.0))
        model.patches.daily_birth_rate[:] = model.scenario.birth_rate_per_day.values
        model.patches.add_vector_property("births", length=model.params.nticks, dtype=np.uint32, default=0)

        return

    def __call__(self, model, tick: int) -> None:
        births = model.prng.poisson(model.patches.daily_birth_rate * model.patches.population[tick, :]).astype(model.patches.births.dtype)
        model.patches.births[tick, :] = births
        model.population.S[tick, :] += births

        return

    def plot(self, fig: Figure = None):
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        # Plot the births ...
        ipatch = self.model.patches.population[:, 0].argmax()
        plt.plot(self.model.patches.births[ipatch, :], label="Births")
        plt.xlabel("Day")
        plt.ylabel("Number of Births")

        yield

        return
