module iLQGames
    using DocStringExtensions
    using Parameters
    using ForwardDiff
    using StaticArrays
    using LinearAlgebra
    import Base:
        getindex,
        setindex!,
        zero

    using Plots
    pyplot()

    # some utils
    include("utils.jl")

    # game utils
    include("strategy.jl")
    include("system_trajectory.jl")
    include("player_cost.jl")

    # dynamics abstraction
    include("control_system.jl")
    include("toy_systems.jl")
    include("linear_system.jl")
    include("product_system.jl")
    include("sim.jl")

    # game abstraction
    include("game.jl")

    # the solver implementations
    include("solve_lq_game.jl")
    include("ilq_solver.jl")

    # some tools for plotting
    include("plot_utils.jl")

end # module
