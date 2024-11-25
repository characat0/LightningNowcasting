#!/bin/bash
sha256sum -c data/SHA256SUMS

# julia --project=. scripts/create_dataset_from_old.jl

julia --project=. scripts/simulate.jl

