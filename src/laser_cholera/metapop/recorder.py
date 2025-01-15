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
        assert hasattr(self.model.agents, "S"), "Recorder: model agents needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.agents, "I"), "Recorder: model agents needs to have a 'I' (infectious) attribute."
        assert hasattr(self.model.agents, "R"), "Recorder: model agents needs to have a 'R' (recovered) attribute."
        assert hasattr(self.model.agents, "V"), "Recorder: model agents needs to have a 'V' (vaccinated) attribute."
        assert hasattr(self.model.agents, "N"), "Recorder: model agents needs to have a 'N' (current agents) attribute."
        assert hasattr(self.model, "patches"), "Recorder: model needs to have a 'patches' attribute."
        assert hasattr(
            self.model.patches, "LAMDA"
        ), "Recorder: model patches needs to have a 'LAMDA' (human-to-human transmission rate) attribute."
        assert hasattr(
            self.model.patches, "PSI"
        ), "Recorder: model patches needs to have a 'PSI' (environment-to-human transmission rate) attribute."
        assert hasattr(self.model.patches, "W"), "Recorder: model patches needs to have a 'W' (environmental contamination) attribute."
        assert hasattr(
            self.model.patches, "beta_env"
        ), "Recorder: model patches needs to have a 'beta_env' (environment-to-human transmission rate) attribute."
        assert hasattr(self.model.patches, "delta"), "Recorder: model patches needs to have a 'delta' (environmental decay rate) attribute."
        # assert hasattr(
        #     self.model.patches, "theta"
        # ), "Recorder: model patches needs to have a 'theta' (environmental decontamination rate) attribute."

        return

    def __call__(self, model, tick: int) -> None:
        # write the current state of the model to an HDF5 file if it is the last tick of the simulation
        if tick == (model.params.nticks - 1):
            filename = Path.cwd() / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.h5"  # noqa: DTZ005
            with h5.File(filename, "w") as file:
                # create a group in the HDF5 file called "agents"
                agents = file.create_group("agents")
                # write the following NumPy arrays to the file: S, I, R, V, N
                agents.create_dataset("S", data=model.agents.S)
                agents.create_dataset("I", data=model.agents.I)
                agents.create_dataset("R", data=model.agents.R)
                agents.create_dataset("V", data=model.agents.V)
                agents.create_dataset("N", data=model.agents.N)
                # create a group in the HDF5 file called "patches"
                patches = file.create_group("patches")
                # write the following NumPy arrays to the file: LAMDA, PSI, W, beta_env, delta, theta
                patches.create_dataset("LAMDA", data=model.patches.LAMDA)
                patches.create_dataset("PSI", data=model.patches.PSI)
                patches.create_dataset("W", data=model.patches.W)
                patches.create_dataset("beta_env", data=model.patches.beta_env)
                patches.create_dataset("delta", data=model.patches.delta)
                # patches.create_dataset("theta", data=model.patches.theta)

        return

    def plot(self, fig: Figure = None):
        yield
        return
