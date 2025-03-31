import gzip
import io
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Union

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

            if True:  # ("compress" in model.params) and model.params.compress:
                save_compressed_hdf5_parameters(model, filename)
            else:
                save_hdf5_parameters(model, filename)

            if self.verbose:
                print(f"Recorder: model state saved to {filename}")

        return

    # def plot(self, fig: Figure = None):  # pragma: no cover
    #     yield
    #     return


def save_hdf5_parameters(model, filename: Union[str, Path]) -> None:
    # Step 1: Write HDF5 content to the file
    with h5.File(filename, "w") as h5file:
        save_hdf5(h5file, model)

    return


def save_compressed_hdf5_parameters(model, filename: Union[str, Path]) -> None:
    # Step 1: Create an in-memory buffer for HDF5
    hdf5_buffer = io.BytesIO()

    # Step 2: Write HDF5 content to the buffer
    with h5.File(hdf5_buffer, "w") as h5file:
        save_hdf5(h5file, model)

    # Step 3: Compress and save to disk
    with gzip.open(filename.with_name(filename.name + ".gz"), "wb") as gz_file:
        gz_file.write(hdf5_buffer.getvalue())

    return


def save_hdf5(h5file: h5.File, model, verbose=False) -> None:
    for frame in ["agents", "patches"]:
        if not hasattr(model, frame):
            raise AttributeError(f"Recorder: model needs to have a '{frame}' attribute.")

        lf = getattr(model, frame)
        properties = dir(lf)
        properties = [p for p in properties if not p.startswith("_")]
        properties = [p for p in properties if not isinstance(getattr(lf, p), (MethodType))]

        group = h5file.create_group(frame)

        for prop in properties:
            if verbose:
                print(f"Recorder: saving {frame}.{prop} ...")
            group.create_dataset(prop, data=getattr(lf, prop))

    return
