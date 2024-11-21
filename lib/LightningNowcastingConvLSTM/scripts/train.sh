#! /bin/bash

sha256sum -c data/SHA256SUMS

julia --project=. -e 'import Pkg; Pkg.instantiate()'

julia --project=. scripts/create_dataset_from_old.jl

julia --project=. scripts/simulate.jl

