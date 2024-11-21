using DrWatson
@quickactivate

# using Lux, Random, Optimisers, Zygote, CUDA, LuxCUDA, NPZ, MLUtils, Printf, ProgressMeter, MLFlowClient, JLD2, Statistics, Plots, Dates, ConvLSTM, Accessors
using MLUtils, JLD2
using ConvLSTM, Lux, CUDA, LuxCUDA, Zygote
using ProgressMeter, MLFlowClient, Plots, Dates, Random, Optimisers, Statistics, DataStructures
using ImageFiltering
import JSON

include(srcdir("metrics.jl"))

const mlf = MLFlow()

const experiment = getorcreateexperiment(mlf, "lux-lightning-4")

const lossfn = BinaryCrossEntropyLoss()


function logsystemmetrics(run_info)
    metrics = Dict(
        :system_memory_usage_megabytes => (Sys.total_memory() - Sys.free_memory()) / (2 ^ 20),
        :system_memory_usage_percentage => (1 - Sys.free_memory()/Sys.total_memory()),
    )
    metrics = Dict(Symbol("system/", k) => v for (k, v) in metrics)
    logmetrics(mlf, run_info.info.run_id, metrics)
end


function apply_bilinearfilter(ds::T)::T where {T}
    k_org = [1 2 1; 2 4 2; 1 2 1]
    k = centered(k_org / Float32(sum(k_org)))
    mapslices(ds, dims=(1, 2)) do chunk
        res = imfilter(chunk, k)
        max.(chunk, res)
    end
end

function apply_gaussian_filter(ds::T, sigma=.9)::T where {T}
    k = Float32.(Kernel.gaussian(sigma))
    mapslices(ds, dims=(1, 2)) do chunk
        res = imfilter(chunk, k)
        max.(chunk, res)
    end
end


function get_dataloaders(batchsize, n_train)
    @load datadir("exp_pro", "train.jld2") dataset
    @show "Loaded training set"
    train = dataset::Array{UInt8, 4} / Float32(typemax(UInt8))
    (x_train, y_train) = reshape(train[:, :, begin:n_train, :], size(train)[1:2]..., 1, n_train, :), train[:, :, 11:20, :]
    # @time "Aplying gaussian filter to y_train" y_train = apply_gaussian_filter(y_train, 1)
    # @time "Applying gaussian filter to x_train" x_train = apply_bilinearfilter(x_train)
    @load datadir("exp_pro", "val.jld2") dataset
    @show "Loaded validation set"
    val = dataset::Array{UInt8, 4} / Float32(typemax(UInt8))
    (x_val, y_val) = reshape(val[:, :, 1:10, :], size(train)[1:2]..., 1, 10, :), val[:, :, 11:20, :]

    return (
        DataLoader((x_train, y_train); batchsize),
        DataLoader((x_val, y_val); batchsize),
    )
end

function plot_predictions(tmp_location, model, train_state, data, run_info, epoch, name="predictions")
    x, y = data
    ps_trained, st_trained = (train_state.parameters, train_state.states)
    ŷ, _ = model(x, ps_trained, Lux.testmode(st_trained))

    for idx in [1, 3, 7]
        data_to_plot = vcat(
            reshape(ŷ[:, :, :, idx], 64, :),
            reshape(y[:, :, :, idx], 64, :),
            reshape(x[:, :, 1, :, idx], 64, :),
        ) |> cpu_device()
        fig = heatmap(data_to_plot, size=(128*10 + 80, 128*3 + 30), clims=(0, 1))
        savefig(fig, "$(tmp_location)/epoch_$(lpad(epoch, 2, '0'))_$(name)_$(idx)_step.png")
        logartifact(mlf, run_info, "$(tmp_location)/epoch_$(lpad(epoch, 2, '0'))_$(name)_$(idx)_step.png")
    end
end

function struct_to_dict(s)
    Dict(fieldnames(typeof(s)) .=> getfield.(Ref(s), fieldnames(typeof(s))))
end


const metrics_to_monitor = Dict(
    lossfn => :loss,
    accuracy => :accuracy,
    f1 => :f1,
)

function evaluate(model, state, metrics, dataset, name)
    metric_values = DefaultDict{Symbol, Float64}(0.0)
    ps, st = state.parameters, Lux.testmode(state.states)
    @time "Evaluation ($(name))" for (x, y) in dataset
        ŷ, st = model(x, ps, st)
        for s in 1:size(y, 3), (func, f) in metrics
            ŷ_t = @view ŷ[:, :, s:s, :]
            y_t = @view y[:, :, s:s, :]

            value = func(ŷ_t, y_t)
            base_key = Symbol(f, "_$(name)")
            metric_values[base_key] += value / (size(y, 3) * length(dataset))
            metric_values[Symbol(base_key, :., lpad(s, 2, '0'))] += value / length(dataset)
        end
    end
    metric_values
end


function simulate(
    run_info;
    device_id=1,
    logging=true,
    mode,
    k_x,
    k_h,
    hidden,
    eta,
    rho,
    batchsize,
    use_bias,
    seed,
    tmp_location = mktempdir(),
    n_steps=30,
)
    @show tmp_location
    dev = gpu_device(device_id, force_gpu_usage=true)
    STEPS_X = 10
    STEPS_Y = 10
    n_train = mode == :conditional ? STEPS_X + STEPS_Y : STEPS_X
    train_loader, val_loader = get_dataloaders(batchsize, n_train) |> dev
    peephole = ntuple(Returns(true), length(use_bias))
    model = SequenceToSequenceConvLSTM((k_x, k_x), (k_h, k_h), 1, hidden, STEPS_X, mode, use_bias, peephole, σ, 1)
    @save "$(tmp_location)/model_config.jld2" model
    logartifact(mlf, run_info, "$(tmp_location)/model_config.jld2")
    rng = Xoshiro(seed)
    ps, st = Lux.setup(rng, model) |> dev
    opt = RMSProp(; eta, rho)
    logparam(mlf, run_info, Dict(
        "model.depth" => length(hidden),
        "model.kernel_hidden" => k_h,
        "model.kernel_input" => k_x,
        "model.hidden_dims" => hidden,
        "model.batchsize" => batchsize,
        "model.use_bias" => use_bias,
        "model.peephole" => peephole,
        "rng.algo" => string(typeof(rng)),
        "rng.seed" => seed,
        "loss.algo" => string(typeof(lossfn)),
        Dict(["rng.$(k)" => v for (k, v) in struct_to_dict(rng)])...,
        "opt.algo" => string(typeof(opt)),
        Dict(["opt.$(k)" => v for (k, v) in struct_to_dict(opt)])...
    ))
    train_state = Training.TrainState(model, ps, st, opt)
    @info "Starting train"
    for epoch in 1:n_steps
        losses = Float32[]
        dt = logging ? 20*60.0 : 0.1
        progress = Progress(length(train_loader); dt=dt, desc="Training Epoch $(epoch)", barlen=32)
        for (x, y) in train_loader
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoZygote(), lossfn, (x, y), train_state
            )
            push!(losses, loss)
            next!(progress; showvalues = [("loss", loss)])
        end
        logmetric(mlf, run_info, "loss_train", mean(losses); step=epoch)
        # Validation run
        metrics_tests = evaluate(model, train_state, metrics_to_monitor, val_loader, :val)
        logmetrics(mlf, run_info.info.run_id, metrics_tests, step=epoch)
        logsystemmetrics(run_info)

        if ((epoch - 1) % 4 == 0) || (epoch == n_steps) 
            ps_trained, st_trained = (train_state.parameters, train_state.states) |> cpu_device()
            @save "$(tmp_location)/trained_weights_$(lpad(epoch, 2, '0')).jld2" ps_trained st_trained
            logartifact(mlf, run_info, "$(tmp_location)/trained_weights_$(lpad(epoch, 2, '0')).jld2")
            plot_predictions(tmp_location, model, train_state, first(val_loader), run_info, epoch, "predictions")
        end

    end
end


function simulate(; kwargs...)
    d = tag!(Dict{Symbol, Any}(), storepatch=true, commit_message=true)
    tags = [
        Dict("key" => "mlflow.source.git.commit", "value" => string(chopsuffix(d[:gitcommit], "-dirty"))),
        Dict("key" => "mlflow.source.name", "value" => get(ENV, "REPOSITORY_URL", "https://github.com/characat0/LightningNowcasting.git")),
        Dict("key" => "mlflow.source.type", "value" => "git"),
        Dict("key" => "mlflow.source.branch", "value" => "main"),
        Dict("key" => "mlflow.note.content", "value" => "Commit message: $(string(d[:gitmessage]))"),
        Dict("key" => "is_dirty", "value" => string(endswith(d[:gitcommit], "-dirty"))),
    ]
    run_info = createrun(mlf, experiment; tags=tags)
    @show run_info.info.run_name
    tmpfolder = mktempdir()
    logging = parse(Bool, get(ENV, "JULIA_SLOW_PROGRESS_BAR", "false"))

    try
        logsystemmetrics(run_info)
        gpu_info = JSON.parse(get(ENV, "GPU_INFO", "{}"))
        if length(gpu_info) > 0
            @show gpu_info
            logparam(mlf, run_info, gpu_info)
        end
    
        if haskey(d, :gitpatch)
            f = "$(tmpfolder)/head.patch"
            write(f, d[:gitpatch])
            logartifact(mlf, run_info, f)
        end
        simulate(run_info; logging=logging, kwargs...)
        updaterun(mlf, run_info, "FINISHED")
    catch e
        if typeof(e) <: InterruptException
            updaterun(mlf, run_info, "KILLED"; end_time=string(Int(trunc(datetime2unix(now(UTC)) * 1000))))
        else
            f = "$(tmpfolder)/error.log"
            write(f, sprint(showerror, e))
            logartifact(mlf, run_info, f)
            updaterun(mlf, run_info, "FAILED"; end_time=string(Int(trunc(datetime2unix(now(UTC)) * 1000))))
        end
        rethrow()
    end

end

h = 64
eta = 3e-3
b = (false, true, true)

simulate(;
    k_h=5,
    k_x=5,
    hidden=(h, h÷2, h÷2),
    seed=42,
    eta=eta,
    rho=0.9,
    n_steps=20,
    batchsize=16,
    use_bias=b,
    mode=:conditional,
)

