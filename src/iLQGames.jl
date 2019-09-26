module iLQGames
    using DocStringExtensions
    using Parameters
    using ForwardDiff
    using DiffResults
    using StaticArrays
    using LinearAlgebra
    using Parameters
    using Colors
    import Base:
        getindex,
        setindex!,
        zero,
        copy

    using Plots
    gr()

    # some useful type shorthands
    const SIndex{N} = Tuple{Vararg{SVector{N, Int}}} where N

    # some macro sugar to make life easier
    include("sugar.jl")

    # game utils
    include("strategy.jl")
    include("system_trajectory.jl")
    include("player_cost.jl")

    # dynamics abstraction
    include("control_system.jl")
    include("linear_system.jl")
    include("product_system.jl")

    # game abstraction
    include("game.jl")

    # the solver implementations
    include("solve_lq_game.jl")
    include("ilq_solver.jl")

    # simulation
    include("sim.jl")

    # some handy tools for problem description
    include("cost_design_utils.jl")

    # some toy examples to work with
    include("toy_systems.jl")
    include("two_player_car_game.jl")

    # some tools for plotting
    include("plot_utils.jl")

end # module
