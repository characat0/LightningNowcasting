using DrWatson
@quickactivate

using Plots, JLD2, Random, CodecZlib

@load datadir("exp_pro", "train.jld2") dataset

rng = Xoshiro(42)
n_samples = 4

samples_axis = 4
time_axis = 3


samples = rand(rng, 1:size(dataset, samples_axis), n_samples)
steps = size(dataset, time_axis)

for s in samples
    anim = @animate for i in 1:steps
        heatmap(dataset[:, :, i, s], clim=(0, 255), yflip=true)
    end
    gif(anim, plotsdir("moving_mnist_$(s).gif"), fps=4)
end

@load datadir("exp_pro", "ood_val.jld2") dataset

samples = rand(rng, 1:size(dataset, samples_axis), n_samples)

for s in samples
    anim = @animate for i in 1:steps
        heatmap(dataset[:, :, i, s], clim=(0, 255), yflip=true)
    end
    gif(anim, plotsdir("moving_mnist_out_of_domain_$(s).gif"), fps=4)
end