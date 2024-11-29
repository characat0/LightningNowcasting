# FlashRecords.jl

using NCDatasets
using Dates

"""
    FlashRecords(ds::NCDataset)
Construye `FlashRecords` a partir de un [NCDatasets.NCDataset](https://alexander-barth.github.io/NCDatasets.jl/latest/dataset/#NCDatasets.NCDataset)
Un [Flash](https://vlab.noaa.gov/web/geostationary-lightning-mapper/products) está fuertemente asociado con la evolución de tormentas. 
Esta estructura agrupa varios Flashes y almacena sus atributos.
# Campos
- `latitude` : representa las latitudes de cada flash en `Float32`
- `longitude` : representa las longitudes de cada flash en `Float32`
- `quality` : flag que indica el estado del Flash #TODO documentar más
- `energy` : cantidad de energía en Joules, puede contener valores nulos
- `area` : tamaño del área
- `time_start` : tiempo de inicio del escaneo
- `time_end` : tiempo de finalización del escaneo
- `dataset_name` : identificador del dataset origen de los Flashes
"""
struct FlashRecords
  latitude::Vector{Float32}
  longitude::Vector{Float32}
  quality::Vector{Bool}
  energy::Vector{Union{Missing, Float32}}
  area::Vector{Union{Missing, Float32}}
  time_start::DateTime
  time_end::DateTime
  dataset_name::String
  function FlashRecords(ds::NCDataset)
    new(
      ds["flash_lat"][:],
      ds["flash_lon"][:],
      ds["flash_quality_flag"][:] .== zero(Int16),
      ds["flash_energy"][:],
      ds["flash_area"][:],
      DateTime(ds.attrib["time_coverage_start"], dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
      DateTime(ds.attrib["time_coverage_end"], dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
      ds.attrib["dataset_name"]
    )
  end
end