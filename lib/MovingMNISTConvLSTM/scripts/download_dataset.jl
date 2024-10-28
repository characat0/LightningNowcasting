using DrWatson

@quickactivate

using ProgressMeter, Downloads

dataset_url = "https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"


N = 1000
p = Progress(N; dt=2.0, barlen=64, desc="Downloading mnist_test_seq.npy")

path = Downloads.download(dataset_url, datadir("exp_raw", "mnist_test_seq.npy"), progress=(total, now) -> (total > 0 && total != now) ? update!(p, (now*N)÷total) : nothing)
finish!(p)

@info "dataset downloaded into $(path)"

using NPZ, MLUtils, JLD2

ds = npzread(datadir("exp_raw", "mnist_test_seq.npy"))::Array{UInt8, 4} / Float32(typemax(UInt8))
ds = permutedims(ds, (3, 4, 1, 2)) # Widht Height Time N-samples

train, val = splitobs(ds; at=0.8)

@save datadir("exp_pro", "mnist_")

