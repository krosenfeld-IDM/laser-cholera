import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Census:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "patches"), "Census: model needs to have a 'patches' attribute."
        model.patches.add_vector_property("N", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(self.model, "params"), "Census: model needs to have a 'params' attribute."

        return

    def check(self):
        """
        check() gets called after all components have been instantiated and initialized.
        At this point, we will dynamically generate the update() method based on which compartments we find in the model.
        """
        prolog = "def update(patches, agents, tick):\n    N_next = patches.N[tick + 1]\n"
        epilog = "    return\n"

        body = ""
        for compartment in ["S", "E", "Isym", "Iasym", "R", "V1", "V2"]:
            if hasattr(self.model.agents, compartment):
                body += f"    {compartment}_next = agents.{compartment}[tick + 1]\n"
                body += f"    N_next[:] += {compartment}_next\n"

        code = compile(
            prolog + body + epilog,
            filename="<string>",
            mode="exec",
        )
        namespace = {}
        exec(code, namespace)  # noqa: S102
        self.update = staticmethod(namespace["update"])

        # Makes me feel a little slimy, but we need to set the initial population size
        self.update(self.model.patches, self.model.agents, -1)

        return

    def __call__(self, model, tick: int) -> None:
        self.update(model.patches, model.agents, tick)

        assert np.all(model.patches.N[tick + 1] >= 0), "N' should not go negative"

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Census (Total Population)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.N[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Total Population")
        plt.legend()

        yield "Census (Total Population)"
        return
