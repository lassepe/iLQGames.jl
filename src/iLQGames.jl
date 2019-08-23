module iLQGames
    using DocStringExtensions
    using ForwardDiff
    using StaticArrays

    include("control_system_interface.jl")
    include("linear_system.jl")
    include("control_system_convenience.jl")

    include("quadratic_cost.jl")
    include("finite_horizon_lq_game.jl")
    include("solve_lq_game.jl")

    include("toy_systems.jl")
    include("utils.jl")

end # module
