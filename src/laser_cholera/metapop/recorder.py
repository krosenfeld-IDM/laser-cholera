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
        assert hasattr(self.model.agents, "Isym"), "Recorder: model agents needs to have a 'Isym' (infectious - symptomatic) attribute."
        assert hasattr(self.model.agents, "Iasym"), "Recorder: model agents needs to have a 'Iasym' (infectious - asymptomatic) attribute."
        assert hasattr(self.model.agents, "R"), "Recorder: model agents needs to have a 'R' (recovered) attribute."
        assert hasattr(self.model.agents, "V1imm"), (
            "Recorder: model agents needs to have a 'V1imm' (vaccinated - one dose, immune) attribute."
        )
        assert hasattr(self.model.agents, "V1sus"), (
            "Recorder: model agents needs to have a 'V1sus' (vaccinated - one dose, susceptible) attribute."
        )
        assert hasattr(self.model.agents, "V1inf"), (
            "Recorder: model agents needs to have a 'V1inf' (vaccinated - one dose, infected) attribute."
        )
        assert hasattr(self.model.agents, "V2imm"), (
            "Recorder: model agents needs to have a 'V2imm' (vaccinated - two dose, immune) attribute."
        )
        assert hasattr(self.model.agents, "V2sus"), (
            "Recorder: model agents needs to have a 'V2sus' (vaccinated - two dose, susceptible) attribute."
        )
        assert hasattr(self.model.agents, "V2inf"), (
            "Recorder: model agents needs to have a 'V2inf' (vaccinated - two dose, infected) attribute."
        )
        assert hasattr(self.model, "patches"), "Recorder: model needs to have a 'patches' attribute."
        assert hasattr(self.model.patches, "N"), "Recorder: model agents needs to have a 'N' (current agents) attribute."
        assert hasattr(self.model.patches, "Lambda"), (
            "Recorder: model patches needs to have a 'Lambda' (human-to-human transmission rate) attribute."
        )
        assert hasattr(self.model.patches, "Psi"), (
            "Recorder: model patches needs to have a 'Psi' (environment-to-human transmission rate) attribute."
        )
        assert hasattr(self.model.patches, "W"), "Recorder: model patches needs to have a 'W' (environmental contamination) attribute."
        assert hasattr(self.model.patches, "beta_env"), (
            "Recorder: model patches needs to have a 'beta_env' (environment-to-human transmission rate) attribute."
        )
        assert hasattr(self.model.patches, "delta_jt"), (
            "Recorder: model patches needs to have a 'delta_jt' (environmental decay rate) attribute."
        )

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
                agents.create_dataset("Isym", data=model.agents.Isym)
                agents.create_dataset("Iasym", data=model.agents.Iasym)
                agents.create_dataset("R", data=model.agents.R)
                agents.create_dataset("V1imm", data=model.agents.V1imm)
                agents.create_dataset("V1sus", data=model.agents.V1sus)
                agents.create_dataset("V1inf", data=model.agents.V1inf)
                agents.create_dataset("V2imm", data=model.agents.V2imm)
                agents.create_dataset("V2sus", data=model.agents.V2sus)
                agents.create_dataset("V2inf", data=model.agents.V2inf)
                # create a group in the HDF5 file called "patches"
                patches = file.create_group("patches")
                # write the following NumPy arrays to the file: Lambda, Psi, W, beta_env, delta, theta
                patches.create_dataset("N", data=model.patches.N)
                patches.create_dataset("non_disease_deaths", data=model.patches.non_disease_deaths)
                patches.create_dataset("disease_deaths", data=model.patches.disease_deaths)
                patches.create_dataset("Lambda", data=model.patches.Lambda)
                patches.create_dataset("Psi", data=model.patches.Psi)
                patches.create_dataset("W", data=model.patches.W)
                patches.create_dataset("beta_env", data=model.patches.beta_env)
                patches.create_dataset("delta_jt", data=model.patches.delta_jt)

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        yield
        return
