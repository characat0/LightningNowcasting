# l2_lcfa_merge.jl

using ProgressMeter
using OrderedCollections

include("./flashrecords.jl")

function maybereaddir(dirpath; join)
  endswith(dirpath, ".nc") && return [dirpath]
  readdir(dirpath; join)
end


function consolidate_folder_lfca(folder, depth)
  files = readdir(folder; join=true)
  for _ in 1:depth
    files = collect(Iterators.flatten(maybereaddir.(files; join=true)))
  end
  filter!(endswith(".nc"), files)
  flashes = FlashRecords[]
  sizehint!(flashes, length(files))
  @showprogress "Progress for $(basename(folder)): " for file in files
    NCDataset(file, "r") do ds
      push!(flashes, FlashRecords(ds))
    end
  end
  flashes
end


function consolidate_lcfa(basepath, depth::Int, bottom_depth::Int = 2)
  subpaths = readdir(basepath; join=true) # Expect to be years
  for _ in 1:depth
    subpaths = collect(Iterators.flatten(readdir.(subpaths; join=true)))
  end
  response = Dict{String, Vector{FlashRecords}}()
  for folder in subpaths
    @info "Consolidating folder $folder"
    response[basename(folder)] = consolidate_folder_lfca(folder, bottom_depth-depth)
  end
  response
end

function consolidate_lcfa(basepath, lvl::Symbol)
  depth = 0
  if lvl === :year
    depth = 0
  elseif lvl === :day
    depth = 1
  elseif lvl === :hour
    depth = 2
  else
    error("Invalid value for lvl, expected (:year, :day, :hour) but received :$lvl")
  end
  consolidate_lcfa(basepath, depth)
end


# fed_grid.jl

using Dates
using NCDatasets
using StatsBase
using Unitful
using Statistics
using ClimateBase
using OrderedCollections
using SplitApplyCombine

# include(srcdir("dataset", "l2_lcfa_merge.jl"))

lat_degree_to_km(::Quantity) = 110.540f0u"km/°"
lon_degree_to_km(degrees::Quantity) = 111.32f0*cos(degrees)u"km/°"

km_to_lat_degree(::Quantity) = (1.f0/110.54f0)u"°/km"
km_to_lon_degree(degrees::Quantity) = (1.f0/lon_degree_to_km(degrees))

# measures taken from https://es.wikipedia.org/wiki/Geograf%C3%ADa_del_Per%C3%BA#Puntos_extremos
const PERU_N = (-3//100)u"°"
const PERU_S = (-16517//900)u"°"
const PERU_E = (-27463//400)u"°"
const PERU_W = (-11711//144)u"°"

StatsBase.push!(h::Histogram,::NTuple,::Missing) = h

function gridrange(resolution::Quantity, N, S, E, W)
  lat_degrees = km_to_lat_degree(resolution) * resolution
  lat_degrees = uconvert(u"°", lat_degrees)
  lat_range = S:lat_degrees:N
  lon_degrees = mean(km_to_lon_degree.(lat_range)) * resolution
  lon_degrees = uconvert(u"°", lon_degrees)
  lat_range_val = Float32(S.val):lat_degrees.val:Float32(N.val)
  lon_range_val = Float32(W.val):lon_degrees.val:Float32(E.val)
  (lon_range_val, lat_range_val)
end

function generate_grid_lat_lon(resolution::Quantity, N=PERU_N, S=PERU_S, E=PERU_E, W=PERU_W)
  lon_range_val, lat_range_val = gridrange(resolution, N, S, E, W)
  Histogram((lon_range_val, lat_range_val), Float32)
end

function groupby_floor(records::AbstractVector{FlashRecords}, p::Period)
  g = group(x -> Dates.floor(x.time_start, p), records)
  OrderedDict(sort(collect(zip(keys(g), values(g))), by = first))
end

function try_regularize_time(time_range)
  ini = first(time_range)
  stp = time_range[2] - ini
  lst = last(time_range)
  rng = ini:stp:lst
  if collect(rng) == time_range
    return rng
  else
    return time_range
  end
end

# Receives a list of records, and return a climate array representing the FED
# It's possible to specify the corners of the grid and both spatial and temporal
# resolution

"""
    generate_climarray(records::AbstractVector{FlashRecords},
                       s::Quantity, 
                       t::Period;
                       N=PERU_N,
                       S=PERU_S,
                       E=PERU=E,
                       W=PERU_W)
Crea un `ClimArray` a partir de los [`FlashRecords`](@ref) recibidos.
# Argumentos
- `records` : lista de `FlashRecords` a agrupar
- `s::Quantity` : resolución espacial del arreglo resultante
- `t::Period` : resolución temporal del arreglo resultante
- `N`, `S`, `E`, `W` : bordes para generar la grilla de acumulación de Flashes

# Detalle
Se calcula el tamaño del arreglo resultante. 
Una grilla 3D rectangular `Float32` es creada con esquinas definidas por `N`, `S`, `E`, `W` (por defecto los bordes del mapa peruano).
Se acumula la ocurrencia de Flashes en la grilla, finalmente es transformada en un `ClimArray` con nombre "FED".

"""
function generate_climarray(records::AbstractVector{FlashRecords}, s::Quantity, t::Period; N=PERU_N, S=PERU_S, E=PERU_E, W=PERU_W)::ClimArray
  groups = groupby_floor(records, t)
  groups_keys = collect(keys(groups))
  t = Minute(t)
  time_edge = Dates.value.(first(groups_keys):t:(last(groups_keys)+t))
  time_range = first(groups_keys):t:last(groups_keys)
  missing_times = sort!(map(x -> findfirst(isequal(x), time_range), collect(setdiff(Set(time_range), Set(groups_keys)))))
  lon_range, lat_range = gridrange(s, N, S, E, W)
  grid = Histogram((lon_range, lat_range, time_edge), Float32)
  for (t, group) in groups
    for r in group
      append!(grid, (r.longitude, r.latitude, fill(Dates.value(t), length(r.latitude))))
    end
    # view(grid.weights, :, :, searchsortedlast(time_edge, Dates.value(t))) ./= length(group)
  end
  lon_dim = Lon(lon_range[begin:end-1]; metadata=ClimateBase.Metadata(
    "units"         => "degrees_east",
    "standard_name" => "longitude",
    "valid_range"   => Float32[-180.0, 360.0],
    "original_range" => Float32[lon_range[1], step(lon_range), lon_range[end-1]+eps(Float32)], # For numeric rounding error
  ))
  lat_dim = Lat(lat_range[begin:end-1]; metadata=ClimateBase.Metadata(
    "units"         => "degrees_north",
    "standard_name" => "latitude",
    "valid_range"   => Float32[-90.0, 90.0],
    "original_range" => Float32[lat_range[1], step(lat_range), lat_range[end-1]+eps(Float32)], # For numeric rounding error
  ))
  time_dim = Ti(time_range; metadata=ClimateBase.Metadata(
    "original_step_minutes" => t.value,
    "missing_indexes" => Vector{Int}(missing_times),
    "units" => "days since 0000-00-01 00:00:00",
    "standard_name" => "time",
    ))
  attrib = Dict(
    "start_time" => string(minimum([x.time_start for x in records])),
    "end_time" => string(maximum([x.time_start for x in records])),
    "spatial_resolution" => string(s),
    "temporal_resolution" => string(t),
  )
  ClimArray(grid.weights, (lon_dim, lat_dim, time_dim); name="FED", attrib=attrib)
end


# Adapted from https://github.com/JuliaClimate/ClimateBase.jl/blob/35b5e8f85638b7f1d3127b7a446de38afba2c6b6/src/io/netcdf_write.jl#L40
"""
    ncwrite_compressed(file::String, Xs; globalattr = Dict(), deflatelevel::Int=1)
Adaptado de [`ClimateBase.ncwrite`](https://juliaclimate.github.io/ClimateBase.jl/dev/netcdf/#ClimateBase.ncwrite) y permite especificar un nivel de compresión. 
"""
function ncwrite_compressed(file::String, Xs; globalattr = Dict(), deflatelevel)
  if any(X -> hasdim(X, Coord), Xs)
    error("""
    Outputing `UnstructuredGrid` coordinates to .nc files is not yet supported,
    but it is an easy fix, see source of `ncwrite`.
    """)
  end

  ds = NCDataset(file, "c"; attrib = globalattr)
  try
    for (i, X) in enumerate(Xs)
      n = string(X.name)
      if n == ""
        n = "x$i"
        @warn "$i-th ClimArray has no name, naming it $(n) instead."
      end
      ClimateBase.add_dims_to_ncfile!(ds, dims(X))
      attrib = X.attrib
      isnothing(attrib) && (attrib = Dict())
      dnames = ClimateBase.dim_to_commonname.(dims(X))
      data = Array(X)
      ClimateBase.NCDatasets.defVar(ds, n, data, (dnames...,); attrib, deflatelevel)
    end
  finally
    close(ds)
  end
end

function ncwrite_compressed(file::String, X::ClimArray; globalattr = Dict(), deflatelevel=1)
  ncwrite_compressed(file, (X,); globalattr, deflatelevel)
end

"""
    read_fed(file::AbstractString)
Regenera el `ClimArray` producido por [`generate_climarray`](@ref) almacenado en formato HDF5.
"""
function read_fed(file)
  fed = ncread(file, "FED")
  lon_dim, lat_dim, time_dim = [x.val for x in dims(fed)]
  min_step = Minute(pop!(time_dim.metadata, "original_step_minutes")::Int32)
  time_range = minimum(time_dim):min_step:maximum(time_dim)
  time_dim2 = Ti(time_range; metadata=time_dim.metadata)
  original_lon_range = pop!(lon_dim.metadata, "original_range")
  #lon_dim2 = Lon(original_lon_range[1]:original_lon_range[2]:original_lon_range[3]; metadata=lon_dim.metadata)
  lon_dim2 = Lon(range(original_lon_range[1], original_lon_range[3], length(lon_dim)); metadata=lon_dim.metadata)
  original_lat_range = pop!(lat_dim.metadata, "original_range")
  #lat_dim2 = Lat(original_lat_range[1]:original_lat_range[2]:original_lat_range[3]; metadata=lat_dim.metadata)
  lat_dim2 = Lat(range(original_lat_range[1], original_lat_range[3], length(lat_dim)); metadata=lat_dim.metadata)
  ClimArray(fed, (lon_dim2, lat_dim2, time_dim2))
end
