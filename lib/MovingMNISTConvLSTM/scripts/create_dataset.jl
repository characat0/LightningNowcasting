using DrWatson
@quickactivate

include(srcdir("dataset_utils.jl"))

using GZip, CodecZlib, Random, JLD2

const MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
const MNIST_TRAIN_DATA_FILENAME = "train-images-idx3-ubyte.gz"
const TRAIN_EXAMPLES = 60_000
const MNIST_IMAGE_SIZE = 28

function take_step!(v, pos)
    v_x, v_y = v
    x, y = pos
    # Take step in the direction
    x, y = pos .+ v

    # Bounce in x
    x, v_x = ifelse(x <= 0, (zero(x), -v_x), (x, v_x))
    x, v_x = ifelse(x >= 1, ( one(x), -v_x), (x, v_x))

    # Bounce in y
    y, v_y = ifelse(y <= 0, (zero(y), -v_y), (y, v_y))
    y, v_y = ifelse(y >= 1, ( one(y), -v_y), (y, v_y))
    (v_x, v_y), (x, y)
end

function get_random_trajectory(rng, steps, speed=0.1)
    x, y = zeros(steps), zeros(steps)
    x[1], y[1], θ = rand(rng, 3)
    v = sincospi(2*θ) .* speed
    for i = 2:steps
        v, (x[i], y[i]) = take_step!(v, (x[i-1], y[i-1]))
    end
    x, y
end

function image_in_grid!(grid, img, pos)
    i_s, j_s = floor.(Int, (size(grid) .- size(img)) .* pos) .+ 1
    i_e, j_e = (i_s, j_s) .+ size(img) .- 1
    grid[i_s:i_e, j_s:j_e] = max.(grid[i_s:i_e, j_s:j_e], img)
    grid
end

function sequence_in_grid!(grid, img, xs, ys)
    for (i, pos) in enumerate(zip(xs, ys))
        image_in_grid!(view(grid, :, :, i), img, pos)
    end
    grid
end

function create_dataset(rng, grid_size, steps, samples, images, n_images=2)
    grid = zeros(UInt8, grid_size, grid_size, steps, samples)
    for i in 1:samples
        for _ in 1:n_images
            img = images[:, :, rand(rng, 1:TRAIN_EXAMPLES)]
            x, y = get_random_trajectory(rng, steps)
            sequence_in_grid!(view(grid, :, :, :, i), img, x, y)
        end
    end
    grid
end


mnist_images = download_with_progress("$(MNIST_URL)$(MNIST_TRAIN_DATA_FILENAME)", datadir("exp_raw", MNIST_TRAIN_DATA_FILENAME))

const images = GZip.open(mnist_images, "rb") do io
    seek(io, 16) # header
    matrices = reshape(read(io, MNIST_IMAGE_SIZE*MNIST_IMAGE_SIZE*TRAIN_EXAMPLES), MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, TRAIN_EXAMPLES)
    permutedims(matrices, (2, 1, 3)) # Flip image
end # Widht Height N-samples

rng = Xoshiro(42)

dataset = create_dataset(rng, 64, 20, 10_000, images, 2)

@save datadir("exp_pro", "train.jld2") dataset
