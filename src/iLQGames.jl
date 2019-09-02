module iLQGames
    using DocStringExtensions
    using ForwardDiff
    using StaticArrays
    using LinearAlgebra
    import Base:
        getindex,
        zero

    include("affine_strategy.jl")
    include("control_system.jl")
    include("linear_system.jl")
    include("player_cost.jl")
    include("quadratic_player_cost.jl")
    include("finite_horizon_lq_game.jl")
    include("solve_lq_game.jl")
    include("toy_systems.jl")
    include("utils.jl")
end # module
