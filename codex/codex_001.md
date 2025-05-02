**user**
This repository implements a meta-population model of Cholera transmission (see the code in src/laser_cholera/metapop particularly). I would like to
 be able to abstract the implementation here so that I could create, leveraging the code that has already been done here, other metapopulation based
 models of disease transmission (for example, measles). In the end I want to create a totally separate module "laser_metapop". For this goal please follow these instructions to copy 
 file / objects that I will need for my generic core (put in src/laser_metapop):

    ────────────────────────────────────────────────────────────────────────
    A.  Files / objects you will want to KEEP and place in a generic core
    ────────────────────────────────────────────────────────────────────────

        1. src/laser_cholera/metapop/model.py            →  simulation engine
           • the Model class, the event loop, metrics collection and plotting stubs
        2. The “component” classes and their pattern
           Susceptible.py, Exposed.py, Infectious.py, Recovered.py, Vaccinated.py
           HumanToHuman.py, EnvToHuman*.py, Environmental.py, Census.py, Recorder.py, …
           • These demonstrate the *contract* each component obeys:
           ─ __init__(self, model) pushes its properties into LaserFrame
           ─ check(self) validates dependencies
           ─ __call__(self, model, tick) advances one time step
           ─ optional plot(self) returns generators of Figures
        3. laser_core.* (external dependency)
           LaserFrame, PropertySet, and PRNG wrappers are already nicely neutral – no change.
        4. src/laser_cholera/metapop/utils.py
           • get_pi_from_lat_long                           (gravity / mobility)
           • fourier_series_double & get_daily_seasonality  (seasonality helpers)
           • override_helper                                (CLI → typed params)
        5. Parameter loader/validator in metapop/params.py
           • The machinery that turns JSON/HDF5 into a PropertySetEx and does dtype
             normalisation is disease-agnostic; only the *schema* is cholera-specific.
        6. Scenario, Analyzer, Recorder, DerivedValues
           • Useful, but can stay in the cholera package and be re-implemented per
             disease; nothing here blocks abstraction.

**codex**
    The new, disease-agnostic core module has been introduced at

        src/laser_metapop/

    Key highlights

    • src/laser_metapop/init.py

        * Re-exports all generic artefacts so downstream models can simply
          `from laser_metapop import Model, get_parameters, …`.

    • src/laser_metapop/model.py

        * Stand-alone simulation engine (Model class, event loop, metrics &
          plotting stubs).
        * All Cholera-specific imports were removed; the engine now accepts an
          arbitrary component list supplied via `Model.components`.

    • src/laser_metapop/params.py

        * Unmodified loader/serialiser machinery (JSON / HDF5) migrated;
        * Validation stub left intentionally empty so each disease can
          implement its own schema checks;
        * Convenience `Parameters` component retained for plotting.

    • src/laser_metapop/utils.py

        * Gravity mobility, Fourier-seasonality helpers and CLI override
          utility copied verbatim (these were already pathogen-neutral).

    These files are identical (or functionally identical where imports were
    stripped) to their originals in laser_cholera.metapop, giving you a
    clean foundation for creating other disease-specific meta-population
    models (e.g. measles) while the existing Cholera package continues to
    work unchanged.