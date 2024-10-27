using MLUtils, NPZ


function get_dataloaders(batchsize, cutoff)
    ds = npzread("mnist_test_seq.npy")::Array{UInt8, 4} / Float32(typemax(UInt8))
    # WHTN
    ds = permutedims(ds, (3, 4, 1, 2))
    ds_x = reshape(ds[:, :, 1:cutoff, :], (size(ds)[1:2]..., 1, cutoff, :))
    ds_y = ds[:, :, 11:20, :]
    @show size(ds_x)
    @show size(ds_y)

    (x_train, y_train), (x_val, y_val) = splitobs((ds_x, ds_y); at=0.8)
    x_val = x_val[:, :, :, 1:10, :]

    return (
        DataLoader((x_train, y_train); batchsize),
        DataLoader((x_val, y_val); batchsize),
    )
end

