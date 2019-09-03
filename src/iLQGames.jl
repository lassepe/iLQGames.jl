module iLQGames
    using DocStringExtensions
    using ForwardDiff
    using StaticArrays
    using LinearAlgebra
    import Base:
        getindex,
        zero

    using Plots
    gr()


    # game utils
    include("strategy.jl")
    include("player_cost.jl")

    # dynamics abstraction
    include("control_system.jl")
    include("linear_system.jl")
    include("product_system.jl")
    # example systems
    include("toy_systems.jl")

    # game abstraction
    include("game.jl")

    # the solver implementation
    include("solve_lq_game.jl")

    # some utils
    include("utils.jl")
end # module
