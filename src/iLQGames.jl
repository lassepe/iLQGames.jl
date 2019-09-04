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

    # some utils
    include("utils.jl")

    # game utils
    include("strategy.jl")
    include("player_cost.jl")

    # dynamics abstraction
    include("control_system.jl")
    include("toy_systems.jl")
    include("linear_system.jl")
    include("product_system.jl")

    # game abstraction
    include("game.jl")

    # the solver implementation
    include("solve_lq_game.jl")

    # some tools for plotting
    include("plot_utils.jl")

end # module
