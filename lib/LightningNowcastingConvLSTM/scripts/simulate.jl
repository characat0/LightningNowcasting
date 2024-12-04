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



hyperparams = JSON.parse(get(ENV, "TRAIN_HYPERPARAMETERS", "{}"))
hyperparams = Dict(Symbol(k) => v for (k, v) in hyperparams)

type_converter = [
    (:mode, Symbol),
    (:hidden, (Tuple ∘ JSON.parse)),
    (:use_bias, (Tuple ∘ JSON.parse)),
    (:k_x, Base.Fix1(parse, Int)),
    (:k_h, Base.Fix1(parse, Int)),
    (:seed, Base.Fix1(parse, Int)),
    (:n_steps, Base.Fix1(parse, Int)),
    (:batchsize, Base.Fix1(parse, Int)),
    (:eta, Base.Fix1(parse, Float64)),
    (:rho, Base.Fix1(parse, Float64)),
]

@info "Raw Hyperparameters" hyperparams

for (k, f) in type_converter
    try
        hyperparams[k] = f(hyperparams[k])
    catch e
        @warn e
    end
end


@info "Parsed Hyperparameters" hyperparams


function get_temperature(device::CUDA.NVML.Device=first(CUDA.NVML.devices()))
    temp = Ref{UInt32}()
    NVML.nvmlDeviceGetTemperature(device, CUDA.NVML.NVML_TEMPERATURE_GPU, temp)
    return Int(temp[])
end

function get_power_usage(nvml_device::NVML.Device=first(CUDA.NVML.devices()))
    power = Ref{UInt32}()
    NVML.nvmlDeviceGetPowerUsage(nvml_device, power)
    return round(power[] * 1e-3; digits=2)
end

function get_power_limit(nvml_device::NVML.Device=first(CUDA.NVML.devices()))
    power_limit = Ref{UInt32}()
    NVML.nvmlDeviceGetEnforcedPowerLimit(nvml_device, power_limit)
    return round(power_limit[] * 1e-3; digits=2)
end

function get_gpu_utilization(nvml_device::NVML.Device=first(CUDA.NVML.devices()))
    util = Ref{NVML.nvmlUtilization_t}()
    NVML.nvmlDeviceGetUtilizationRates(nvml_device, util)
    return (compute=Int(util[].gpu), mem=Int(util[].memory))
end

function logsystemmetrics(run_info)
    function get_gpu_utilization(nvml_device::NVML.Device=first(CUDA.NVML.devices()))
        util = Ref{NVML.nvmlUtilization_t}()
        NVML.nvmlDeviceGetUtilizationRates(nvml_device, util)
        return (compute=Int(util[].gpu), mem=Int(util[].memory))
    end
    metrics = Dict{Symbol, Real}(
        :system_memory_usage_megabytes => (Sys.total_memory() - Sys.free_memory()) / (1024 ^ 2),
        :system_memory_usage_percentage => (1 - Sys.free_memory()/Sys.total_memory()),
        :gpu_memory_usage_megabytes => (CUDA.total_memory() - CUDA.free_memory()) / (1024 ^ 2),
        :gpu_memory_usage_percentage => (1 - CUDA.free_memory()/CUDA.total_memory()),

    )
    try #  Get NVML metrics
        metrics[:gpu_power_usage_watts] = get_power_usage()
        metrics[:gpu_power_usage_percentage] = metrics[:gpu_power_usage_watts]/get_power_limit()
        utilization = get_gpu_utilization()
        metrics[:gpu_utilization_percentage] = utilization.compute / 100.0
        metrics[:gpu_utilization_percentage_mem] = utilization.mem / 100.0
    catch
    end

    metrics = Dict(Symbol("system/", k) => v for (k, v) in metrics)
    logmetrics(mlf, run_info.info.run_id, metrics)
end


function get_dataloaders(batchsize, mode)
    @load datadir("exp_pro", "train.jld2") dataset_x dataset_y dataset_y_teaching
    @info "Loaded training set"
    train_x = dataset_x::Array{UInt8, 4} / Float32(typemax(UInt8))
    y_train = dataset_y::Array{UInt8, 4} / Float32(typemax(UInt8))
    if mode == :conditional_teaching
        @info "Conditional teaching enabled"
        dataset_teaching = dataset_y_teaching::Array{UInt8, 4} / Float32(typemax(UInt8))
        train_x = cat(train_x, dataset_teaching; dims=Val(3))
    end
    x_train = reshape(train_x, size(train_x)[1:2]..., 1, size(train_x, 3), :)
    @info "Splitted between x and y"
    
    @load datadir("exp_pro", "val.jld2") dataset_x dataset_y
    @info "Loaded validation set"
    x_dataset = dataset_x::Array{UInt8, 4} / Float32(typemax(UInt8))
    y_dataset = dataset_y::Array{UInt8, 4} / Float32(typemax(UInt8))
    (x_val, y_val) = reshape(dataset_x, size(x_dataset)[1:2]..., 1, 10, :), dataset_y

    @show size(x_train)
    @show size(x_val)
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
    STEPS_Y = 10
    train_loader, val_loader = get_dataloaders(batchsize, mode) |> dev
    peephole = ntuple(Returns(true), length(use_bias))
    model = SequenceToSequenceConvLSTM((k_x, k_x), (k_h, k_h), 1, hidden, STEPS_Y, mode, use_bias, peephole, σ, 1)
    @show typeof(model)
    @save "$(tmp_location)/model_config.jld2" model
    logartifact(mlf, run_info, "$(tmp_location)/model_config.jld2")
    rng = Xoshiro(seed)
    d_rng = Dict(["rng.$(k)" => v for (k, v) in struct_to_dict(rng)])
    ps, st = Lux.setup(rng, model) |> dev
    opt = RMSProp(; eta, rho)
    logparam(mlf, run_info, Dict(
        "model.mode" => mode,
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
        d_rng...,
        "opt.algo" => string(typeof(opt)),
        Dict(["opt.$(k)" => v for (k, v) in struct_to_dict(opt)])...
    ))
    train_state = Training.TrainState(model, ps, st, opt)
    @info "Starting train"
    dt = logging ? 15*60.0 : 1.0
    for epoch in 1:n_steps
        losses = Float32[]
        progress = Progress(length(train_loader); dt=dt, desc="Training Epoch $(epoch)", barlen=32)
        for (x, y) in train_loader
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoZygote(), lossfn, (x, y), train_state
            )
            push!(losses, loss)
            next!(progress; showvalues = [("loss", loss)])
        end
        GC.gc(true)
        CUDA.reclaim()
        # Validation run
        metrics_tests = evaluate(model, train_state, metrics_to_monitor, val_loader, :val)
        logmetrics(mlf, run_info.info.run_id, metrics_tests, step=epoch)
        GC.gc(true)
        if ((epoch - 1) % 3 == 0) || (epoch == n_steps) 
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
        Dict("key" => "mlflow.note.content", "value" => "Commit message: $(string(d[:gitmessage])) \nHyperparameters:\n$(get(ENV, "TRAIN_HYPERPARAMETERS", "{}"))"),
        Dict("key" => "is_dirty", "value" => string(endswith(d[:gitcommit], "-dirty"))),
    ]
    run_info = createrun(mlf, experiment; tags=tags)
    @show run_info.info.run_name
    tmpfolder = mktempdir()
    logging = parse(Bool, get(ENV, "JULIA_SLOW_PROGRESS_BAR", "false"))
    @show get(ENV, "JULIA_SLOW_PROGRESS_BAR", missing)

    try
        timer = Timer(t -> begin
            try
                logsystemmetrics(run_info)
            catch e
                @warn e
            end
        end, 60*5; interval=30)
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
        close(timer)
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




simulate(;
    hyperparams...
)

