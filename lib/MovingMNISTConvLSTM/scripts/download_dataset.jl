using DrWatson

@quickactivate

using ProgressMeter, Downloads

dataset_url = "https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"


N = 1000
p = Progress(N; dt=2.0, barlen=64, desc="Downloading mnist_test_seq.npy")

path = Downloads.download(dataset_url, datadir("exp_raw/mnist_test_seq.npy"), progress=(total, now) -> (total > 0 && total != now) ? update!(p, (now*N)÷total) : nothing)
finish!(p)

@info "dataset downloaded into $(path)"
