#! /bin/bash

sha256sum -c data/SHA256SUMS

/usr/local/julia/bin/julia --project=. -e 'import Pkg; Pkg.instantiate()'

/usr/local/julia/bin/julia --project=. scripts/create_dataset_from_old.jl

/usr/local/julia/bin/julia --project=. scripts/simulate.jl

