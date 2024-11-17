import ConvLSTM

test_path = joinpath(pkgdir(ConvLSTM), "test", "runtests.jl")

include(test_path)

