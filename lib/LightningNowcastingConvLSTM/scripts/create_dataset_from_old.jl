using DrWatson
@quickactivate

using HDF5, JLD2, MLUtils, Random, CodecZlib

fed = h5read(datadir("exp_raw", "dataset.h5"), "FED")
lat = h5read(datadir("exp_raw", "dataset.h5"), "lat")
lon = h5read(datadir("exp_raw", "dataset.h5"), "lon")
time = h5read(datadir("exp_raw", "dataset.h5"), "time")

fed = permutedims(fed[:, :, 1, :, :], (1, 2, 4, 3)) # WxHxTxN

train, val = splitobs(fed; at=.8)

function augment(rng, ds)
    n = size(ds, 4)
    output = zeros(eltype(ds), size(ds)[1:3]..., size(ds, 4)*4)
    idx = shuffle(rng, axes(output, 4))
    for i in 0:3
        @info "rotating $i"
        output[:, :, :, idx[n*i+1:n*(i+1)]] = mapslices(Base.Fix2(rotr90, i), ds, dims=(1, 2))
    end
    output
end

dataset = augment(Xoshiro(42), train)

@save datadir("exp_pro", "train.jld2") {compress=true} dataset lat lon time

dataset = augment(Xoshiro(42), val)

@save datadir("exp_pro", "val.jld2") {compress=true} dataset lat lon time

