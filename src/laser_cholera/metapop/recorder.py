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
            root = Path(model.params.outdir) if "outdir" in model.params and model.params.outdir else Path.cwd()
            root.mkdir(parents=True, exist_ok=True)
            filename = root / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.h5"  # noqa: DTZ005

            if True:  # ("compress" in model.params) and model.params.compress:
                filename = save_compressed_hdf5_parameters(model, filename, model.params.verbose)
            else:
                filename = save_hdf5_parameters(model, filename, model.params.verbose)

            if self.verbose:
                print(f"Recorder: model state saved to {filename}")

        return

    # def plot(self, fig: Figure = None):  # pragma: no cover
    #     yield
    #     return


def save_hdf5_parameters(model, filename: Union[str, Path], verbose: bool = False) -> Path:
    # Step 1: Write HDF5 content to the file
    with h5.File(filename, "w") as h5file:
        save_hdf5(h5file, model, verbose)

    return filename  # Unmodified


def save_compressed_hdf5_parameters(model, filename: Union[str, Path], verbose: bool = False) -> Path:
    # Step 1: Create an in-memory buffer for HDF5
    hdf5_buffer = io.BytesIO()

    # Step 2: Write HDF5 content to the buffer
    with h5.File(hdf5_buffer, "w") as h5file:
        save_hdf5(h5file, model, verbose)

    # Step 3: Compress and save to disk
    filename = filename.with_name(filename.name + ".gz")
    with gzip.open(filename, "wb") as gz_file:
        gz_file.write(hdf5_buffer.getvalue())

    return filename  # With .gz extension


def save_hdf5(h5file: h5.File, model, verbose=False) -> None:
    for frame in ["agents", "patches"]:
        if not hasattr(model, frame):
            raise AttributeError(f"Recorder: model needs to have a '{frame}' attribute.")

        lf = getattr(model, frame)
        properties = dir(lf)

        # TODO - automagically write all properties if a subset is not specified

        properties = [p for p in properties if not p.startswith("_")]
        properties = [p for p in properties if not isinstance(getattr(lf, p), (MethodType))]

        # TODO - if a subset is specified, don't bother with the filtering above

        properties = [
            p
            for p in properties
            if p
            in [
                "disease_deaths",
                "expected_cases",
            ]  # ["S", "E", "Isym", "Iasym", "R", "V1imm", "V1sus", "V1inf", "V2imm", "V2sus", "V2inf", "W", "disease_deaths"]
        ]

        group = h5file.create_group(frame)

        for prop in properties:
            if verbose:
                print(f"Recorder: saving {frame}.{prop} ...")
            group.create_dataset(prop, data=getattr(lf, prop))

    return
