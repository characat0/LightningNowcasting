using Dates
using Unitful
using ConvLSTM
using ConvLSTM.Lux
using ArgParse
using ClimateBase
using JLD2
using Random

include("../src/gdal.jl")
include("../src/flashrecords.jl")
include("../src/dataset.jl")

s = ArgParseSettings()
@add_arg_table s begin
  "--model_config"
    help = "Model config path"
    required = true

  "--model_weights"
    help = "Model weights path"
    required = true

  "--input_dir"
    help = "LCFA files path"
    required = true

  "--output_dir"
    help = "Prediction output directoy"
    range_tester = isdir
    default = "."

  "--start_datetime"
    help = "Start time to collect data"
    default = string(now(UTC))
end

# function Lux.init_rnn_hidden_state(
#   rng::AbstractRNG,
#   lstm::ConvLSTMCell,
#   x::AbstractArray,
# )
#   # TODO: Once we support moving `rng` to the device, we can directly initialize on the
#   #       device
#   N = ndims(x)
#   input_size = ntuple(i -> size(x, i), N - 2)
#   hidden_size =
#       calc_out_dims(input_size, lstm.Wx.pad, lstm.Wx.kernel_size, lstm.Wx.stride)
#   channels = lstm.Wh.in_chs
#   lstm.init_state(rng, hidden_size..., channels, size(x, N)) |> Lux.get_device(x)
# end


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
  @load args[:model_config] model
  @load args[:model_weights] ps_trained st_trained
  st = Lux.testmode(st_trained)
  ps = ps_trained

  spatial_resolution = 4u"km"
  temporal_resolution = Minute(15)
  folder = args[:input_dir] # datadir("exp_raw", "22")
  output_folder = args[:output_dir]

  model_steps = 10

  start = DateTime(args[:start_datetime], dateformat"YYYY-m-dTH:M:S.s") # DateTime(2023, 6, 1)# now(UTC) - Year(1)
  finish = start - temporal_resolution * model_steps # DateTime(2023, 5, 31)# start - temporal_resolution * model_steps

  @info "parameters" start finish model_steps stride temporal_resolution
  flashes = read_flashes(get_files(folder, finish, start))
  @assert length(flashes) > 0 "No data to process"
  climarr = generate_climarray(flashes, spatial_resolution, temporal_resolution)

  X = reshape(climarr.data, size(climarr.data, 1), size(climarr.data, 2), 1, size(climarr.data, 3), 1)
  @show typeof(X)
  @show size(X)

  @time "Predicting" pred, _ = model(X, ps, st) # WHT1
  @assert size(pred, 4) == 1 "4th dimension should be 1"
  future_steps = size(pred, 3)

  time = dims(climarr, Ti)[end]+temporal_resolution:temporal_resolution:dims(climarr, Ti)[end]+temporal_resolution*future_steps

  for (i, t) in enumerate(time)
    result = ClimArray(view(pred, :, :, i, 1), (dims(climarr, Lon), dims(climarr, Lat)), "probability of flash")
    out_file = joinpath(output_folder, "GLM-PREDICTED-$(togoesdate(t)).tif")
    save_tiff_uint8(out_file, result)
  end

end

if abspath(PROGRAM_FILE) == @__FILE__
  args = parse_args(s; as_symbols=true)
  main(args)
end
