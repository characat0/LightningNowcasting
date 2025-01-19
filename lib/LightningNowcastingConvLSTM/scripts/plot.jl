using DrWatson
@quickactivate

using DataFrames
using MLFlowClient
using Plots

const mlf = MLFlow()

const experiment = getorcreateexperiment(mlf, "lux-lightning-4")

const runs_to_compare = [
    "eaa6ad5fde4340d3a76767f654ca3194", # baseline-persistence
    "3bee86babb0b42269b8567699011b702", # selective-mole-678
    "b468b6c56fe743c4a26bc210c172fca2", # defiant-stork-187 
    "fc31eb5335d54893aa64736df61b5638", # clumsy-jay-43
    "9cf5435b2bc8423598857c16c8631711", # powerful-kite-400 
    "7b60daefb4b243cb833b9d3b23a811fb", # capricious-cod-694 
    "9486c6933faa4f8ebeb49918eb95fce6", # lyrical-crow-651 
    "c60023257ccb4cd6a9fb8919da19a0c5", # stylish-grub-622
    "075e4a475f0c4cc8be8c553fe71d39f4", # righteous-auk-438 ⭐️
    "795112b505ad44428969faed897f1809", # bald-sow-202
    "278c6dcdf00745e5bbbf80923cf9b1e7", # orderly-kite-460
]




function get_metric(mlf, run_id, metric)
    return MLFlowClient.mlfget(mlf, "metrics/get-history"; run_id=run_id, metric_key=metric)["metrics"]
end

function get_metrics(mlf, run_id, metrics)
    metrics_values = [
        get_metric(mlf, run_id, m)
        for m in metrics
    ]
    DataFrame(collect(Iterators.flatten(metrics_values)))
end

function get_evolution_best_epoch(run_id, metric, ascending=false, step=nothing)
    best_f = ascending ? argmin : argmax
    mlflow_run = getrun(mlf, run_id)
    run_metrics = mlflow_run.data.metrics
    run_name = mlflow_run.info.run_name
    metrics = filter(startswith("$(metric)."), keys(run_metrics))
    values = MLFlowRunDataMetric.(get_metric(mlf, run_id, metric))
    if !isnothing(step)
        best_metric = values[findfirst(v -> v.step === step, values)]
        @info "$run_name: The value for $(metric) is $(round(best_metric.value, digits=3)) at step $(best_metric.step)"
    else
        best_metric = best_f(Base.Fix2(getfield, :value), values)
        @info "$run_name: The best value for $(metric) is $(round(best_metric.value, digits=3)) at step $(best_metric.step)"
    end
    best_step = best_metric.step
    df = get_metrics(mlf, run_id, metrics)
    evolution = sort(filter(r -> r.step === best_step, df), [:key])[!, :value]
    return ("$(run_name) @$(best_step)", evolution)
end



for r_id in runs_to_compare
    label, series = get_evolution_best_epoch(r_id, "f1_val", false)
    plot!(series, label=label)
end
