using ProgressMeter, Dates, Unitful, Lux, GDAL
using Dates
using Unitful
@time using Flux
@time using GDAL_jll
@time using ArgParse


@time include("../src/dataset.jl")
@time include("../src/utils.jl")
@time include("../src/model.jl")



s = ArgParseSettings()
@add_arg_table s begin
  "--model"
    help = "Model path"
    required = true

  "--input_dir"
    help = "LCFA files path"
    required = true

  "--output_dir"
    help = "Prediction output directoy"
    range_tester = isdir
    default = "."

  "--stride"
    help = "Stride for prediction"
    default = 16
    arg_type = Int
  
  "--start_datetime"
    help = "Start time to collect data"
    default = string(now(UTC))
end





# Fix for Upsample layers not accepting SubArray, instead of view we use 
# getindex
function Flux.eachlastdim(A::AbstractArray{T,N}) where {T,N}
  inds_before = ntuple(_ -> :, N-1)
  return (getindex(A, inds_before..., i) for i in axes(A, N))
end


function togoesdate(t)
  years = lpad(year(t), 4, '0')
  days = lpad(dayofyear(t), 3, '0')
  hours = lpad(hour(t), 2, '0')
  minutes = lpad(minute(t), 2, '0')
  seconds = lpad(second(t), 2, '0')
  tenth = millisecond(t) ÷ 1000
  return "$(years)$(days)$(hours)$(minutes)$(seconds)$(tenth)"
end

function floor_seconds(t, secs)
  stamp = datetime2unix(t)
  unix2datetime(stamp - (stamp % secs))
end

function get_start_date_goes_file(fname)
  bname = basename(fname)
  s = length("OR_GLM-L2-LCFA_G16_s") + 1
  """
  Docs on scan time
  s20171671145342: is start of scan time
  4 digit year
  3 digit day of year
  2 digit hour
  2 digit minute
  2 digit second
  1 digit tenth of second
  """
  l = 4 + 3 + 2 + 2 + 2 + 1
  bname[s:s+l-1]
end

function get_files(root_folder, time_from, time_to)
  # example OR_GLM-L2-LCFA_G16_s20232440903200_e20232440903400_c20232440903418.nc
  # use UTC dates
  files = readdir(root_folder, join=true)
  filter!(f -> occursin("OR_GLM-L2-LCFA_G16", f), files)
  start_time = togoesdate(time_from)
  end_time = togoesdate(time_to)
  @info "Filter files by start time" start_time end_time
  filter!(f -> start_time <= get_start_date_goes_file(f) <= end_time, files)  
end


function read_flashes(files)
  flashes = FlashRecords[]
  @showprogress "Reading flashes" for fname in files
    NCDataset(fname, "r") do ds
      push!(flashes, FlashRecords(ds))
    end
  end
  flashes
end


function pad_matrix(mat, pad_size)
  m,n,t = size(mat)
  padded_input = zeros(Float32, m + 2 * pad_size, n + 2 * pad_size, t)
  padded_input[pad_size+1:pad_size+m, pad_size+1:pad_size+n, :] .= mat
  return padded_input
end


function main(args)
  model = BSON.load(args[:model])[:model]
  spatial_resolution = 4u"km"
  temporal_resolution = Minute(15)
  folder = args[:input_dir] # datadir("exp_raw", "22")
  output_folder = args[:output_dir]

  model_steps = 10
  W = 64
  stride = args[:stride]

  start = DateTime(args[:start_datetime], dateformat"YYYY-m-dTH:M:S.s") # DateTime(2023, 6, 1)# now(UTC) - Year(1)
  finish = start - temporal_resolution * model_steps # DateTime(2023, 5, 31)# start - temporal_resolution * model_steps

  @info "parameters" start finish model_steps stride temporal_resolution
  flashes = read_flashes(get_files(folder, finish, start))
  @assert length(flashes) > 0 "No data to process"
  climarr = generate_climarray(flashes, spatial_resolution, temporal_resolution)

  input_array = Flux.pad_zeros(climarr.data, (W ÷ 2, W ÷ 2, 0))
  m, n, _ = size(input_array)

  pred = zeros(Float32, m, n, model_steps)
  counts = zeros(Float32, m, n)

  @showprogress for (i, j) in Base.Iterators.product(1:stride:m-W, 1:stride:n-W)
    Flux.reset!(model)
    pred[i:i+W-1, j:j+W-1, :] += reshape(
      model(
        reshape(input_array[i:i+W-1, j:j+W-1, :], W, W, 1, 1, :)
      ),
      W, W, :,
    )
    counts[i:i+W-1, j:j+W-1, :] .+= 1
  end
  pred = (pred ./ counts)[W÷2+1:end-W÷2, W÷2+1:end-W÷2, :]
  time = dims(climarr, Ti)[end]+temporal_resolution:temporal_resolution:dims(climarr, Ti)[end]+temporal_resolution*model_steps

  pred_uint = floor.(UInt8, pred * (2^8-1))
  for (i, t) in enumerate(time)
    result = ClimArray(pred_uint[:, :, i], (dims(climarr, Lon), dims(climarr, Lat)), "probability of flash")
    tf = tempname() * ".nc"
    ncwrite(tf, result)
    out_file = joinpath(output_folder, "GLM-PREDICTED-$(togoesdate(t)).tif")
    GDAL_jll.gdal_translate_exe() do exe
      run(`$exe -ot Byte $tf $out_file`)
    end
  end

end

if abspath(PROGRAM_FILE) == @__FILE__
  args = parse_args(s; as_symbols=true)
  main(args)
end