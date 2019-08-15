module iLQGames
    using DocStringExtensions
    using ForwardDiff
    using StaticArrays

    export
        ControlSystem
    include("control_system.jl")

    # some toy systems to play with
    include("toy_systems.jl")
end # module
