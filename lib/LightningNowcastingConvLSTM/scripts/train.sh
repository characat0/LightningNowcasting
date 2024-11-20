#! /bin/bash

sha256sum -c data/SHA256SUMS

julia scripts/create_dataset_from_old.jl

julia scripts/simulate.jl

