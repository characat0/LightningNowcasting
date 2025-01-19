using DrWatson
@quickactivate

# using Lux, Random, Optimisers, Zygote, CUDA, LuxCUDA, NPZ, MLUtils, Printf, ProgressMeter, MLFlowClient, JLD2, Statistics, Plots, Dates, ConvLSTM, Accessors
using MLUtils, JLD2
using Lux
using ProgressMeter, MLFlowClient, Plots, Dates, Random, Optimisers, Statistics, DataStructures
using ImageFiltering
import JSON

include(srcdir("metrics.jl"))

const mlf = MLFlow()

const experiment = getorcreateexperiment(mlf, "lux-lightning-4")

const lossfn = BinaryCrossEntropyLoss()


function get_dataloaders(batchsize)
    @load datadir("exp_pro", "val.jld2") dataset_x dataset_y
    @info "Loaded validation set"
    x_dataset = dataset_x::Array{UInt8, 4} / Float32(typemax(UInt8))
    y_dataset = dataset_y::Array{UInt8, 4} / Float32(typemax(UInt8))
    (x_val, y_val) = reshape(x_dataset, size(x_dataset)[1:2]..., 1, 10, :), y_dataset

    @show size(x_val)
    return DataLoader((x_val, y_val); batchsize)
end

function plot_predictions(tmp_location, model, data, run_info, epoch, name="predictions")
    x, y = data
    ŷ, _ = model(x)

    for idx in [1, 3, 7]
        data_to_plot = vcat(
            reshape(ŷ[:, :, :, idx], 64, :),
            reshape(y[:, :, :, idx], 64, :),
            reshape(x[:, :, 1, :, idx], 64, :),
        ) |> cpu_device()
        fig = heatmap(data_to_plot, size=(100*10 + 80, 100*3 + 30), clims=(0, 1))
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

function evaluate(model, metrics, dataset, name)
    metric_values = DefaultDict{Symbol, Float64}(0.0)
    @time "Evaluation ($(name))" for (x, y) in dataset
        ŷ, _ = model(x)
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

function build_model(steps)
    function model(x::AbstractArray{T, N}) where {T, N}
        last_step = size(x, N-1)
        repeats = ntuple(i -> ifelse(i==N-1, steps, 1), N)
        prev = repeat(selectdim(x, N-1, last_step:last_step), inner=repeats)
        (selectdim(prev, N-2, 1), nothing)
    end
    return model
end


function simulate(
    run_info;
    seed,
    batchsize,
    tmp_location = mktempdir(),
)
    @show tmp_location
    epoch = 1
    STEPS_Y = 10
    model = build_model(STEPS_Y)
    val_loader = get_dataloaders(batchsize)

    rng = Xoshiro(seed)
    d_rng = Dict(["rng.$(k)" => v for (k, v) in struct_to_dict(rng)])
    logparam(mlf, run_info, Dict(
        "model.method" => "persistence",
        "rng.algo" => string(typeof(rng)),
        "rng.seed" => seed,
        d_rng...,
    ))
    metrics_tests = evaluate(model, metrics_to_monitor, val_loader, :val)
    logmetrics(mlf, run_info.info.run_id, metrics_tests, step=epoch)
    plot_predictions(tmp_location, model, first(val_loader), run_info, epoch, "predictions")
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
    @show get(ENV, "JULIA_SLOW_PROGRESS_BAR", missing)

    try
        simulate(run_info; kwargs...)
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
    seed=42,
    batchsize=64,
)

