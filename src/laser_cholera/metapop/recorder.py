from datetime import datetime
from pathlib import Path
from types import MethodType

import h5py as h5


class Recorder:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = model.params.verbose
        return

    def check(self):
        if not hasattr(self.model, "agents"):
            Warning("Recorder: model expected to have a 'agents' attribute.")
        if not hasattr(self.model, "patches"):
            Warning("Recorder: model expected to have a 'patches' attribute.")

        return

    def __call__(self, model, tick: int) -> None:
        # write the current state of the model to an HDF5 file if it is the last tick of the simulation
        if tick == (model.params.nticks - 1):
            filename = Path.cwd() / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.h5"  # noqa: DTZ005
            with h5.File(filename, "w") as file:
                for frame in ["agents", "patches"]:
                    if not hasattr(model, frame):
                        raise AttributeError(f"Recorder: model needs to have a '{frame}' attribute.")
                    lf = getattr(model, frame)
                    properties = dir(lf)
                    properties = [p for p in properties if not p.startswith("_")]
                    properties = [p for p in properties if not isinstance(getattr(lf, p), (MethodType))]

                    # create a group in the HDF5
                    group = file.create_group(frame)

                    for prop in properties:
                        if self.verbose:
                            print(f"Recorder: saving {frame}.{prop} ...")
                        group.create_dataset(prop, data=getattr(lf, prop))

                if self.verbose:
                    print(f"Recorder: model state saved to {filename}")

        return

    # def plot(self, fig: Figure = None):  # pragma: no cover
    #     yield
    #     return
