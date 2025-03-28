from datetime import datetime
from pathlib import Path

import h5py as h5
from matplotlib.figure import Figure


class Recorder:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose
        return

    def check(self):
        assert hasattr(self.model, "agents"), "Recorder: model needs to have a 'agents' attribute."
        assert hasattr(self.model, "patches"), "Recorder: model needs to have a 'patches' attribute."

        return

    def __call__(self, model, tick: int) -> None:
        # write the current state of the model to an HDF5 file if it is the last tick of the simulation
        if tick == (model.params.nticks - 1):
            filename = Path.cwd() / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.h5"  # noqa: DTZ005
            with h5.File(filename, "w") as file:
                # create a group in the HDF5 file called "agents"
                agents = file.create_group("agents")

                for compartment in ["S", "E", "Isym", "Iasym", "R", "V1imm", "V1sus", "V1inf", "V2imm", "V2sus", "V2inf", "V1", "V2"]:
                    if hasattr(model.agents, compartment):
                        agents.create_dataset(compartment, data=getattr(model.agents, compartment))
                    else:
                        if self.verbose:
                            print(f"Recorder: {compartment} not found in model.agents.")

                # create a group in the HDF5 file called "patches"
                patches = file.create_group("patches")

                for trace in [
                    "N",
                    "births",
                    "non_disease_deaths",
                    "disease_deaths",
                    "Lambda",
                    "Psi",
                    "W",
                    "beta_env",
                    "delta_jt",
                    "V1_incidence_env",
                    "V2_incidence_env",
                    "V1_incidence_hum",
                    "V2_incidence_hum",
                ]:
                    if hasattr(model.patches, trace):
                        patches.create_dataset(trace, data=getattr(model.patches, trace))
                    else:
                        if self.verbose:
                            print(f"Recorder: {trace} not found in model.patches.")

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        yield
        return
