FROM julia:1.10.6-bookworm

RUN apt-get update && apt-get install g++ --yes

WORKDIR /app/

COPY Project.toml Manifest.toml ./
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'

COPY . ./

RUN julia --project=. -e 'using PackageCompiler; PackageCompiler.create_sysimage(cpu_target="generic", sysimage_path="inference-sysimage.so", precompile_statements_file="scripts/precompile.jl")'

ENTRYPOINT ["julia", "-Jinference-sysimage.so" , "--project=.", "scripts/inference.jl", "--model=/app/model.bson", "--input_dir=/app/input", "--output_dir=/app/output"]
