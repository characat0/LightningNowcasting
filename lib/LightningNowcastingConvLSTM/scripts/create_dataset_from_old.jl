using DrWatson
@quickactivate

using HDF5, JLD2, MLUtils, Random, CodecZlib
using ImageFiltering

fed = h5read(datadir("exp_raw", "dataset.h5"), "FED")
lat = h5read(datadir("exp_raw", "dataset.h5"), "lat")
lon = h5read(datadir("exp_raw", "dataset.h5"), "lon")
time = h5read(datadir("exp_raw", "dataset.h5"), "time")

fed = convert.(UInt8, fed * Float32(255))

fed = permutedims(fed[:, :, 1, :, :], (1, 2, 4, 3)) # WxHxTxN

train, val = splitobs(fed; at=.97)

function uint8_filter(chunk, k)
    chunk = chunk / Float32(255)
    res = imfilter(chunk, k)
    res .= max.(chunk, res)
    ceil.(UInt8, res * Float32(255))
end

function augment(rng, ds, n_empty=0, gaussian_filter=false)
    n = size(ds, 4)
    output = zeros(eltype(ds), size(ds)[1:3]..., size(ds, 4)*4 + n_empty)
    if gaussian_filter
        @info "applying gaussian filter"
        k = Float32.(Kernel.gaussian(.9))
        ds = mapslices(Base.Fix2(uint8_filter, k), ds, dims=(1, 2))
    end
    idx = shuffle(rng, axes(output, 4))
    for i in 0:3
        @info "rotating $i"
        output[:, :, :, idx[n*i+1:n*(i+1)]] = mapslices(Base.Fix2(rotr90, i), ds, dims=(1, 2))
    end
    output
end

dataset = augment(Xoshiro(42), train, 2_000, true)

@info "number of samples for train: $(size(dataset, 4))"

@save datadir("exp_pro", "train.jld2") {compress=true} dataset lat lon time

dataset = augment(Xoshiro(42), val, 200)

@info "number of samples for validation: $(size(dataset, 4))"

@save datadir("exp_pro", "val.jld2") {compress=true} dataset lat lon time

