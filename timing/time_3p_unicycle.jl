using Test
using BenchmarkTools

using iLQGames:
    iLQSolver,
    Unicycle4D,
    NPlayerUnicycleCost,
    AffineStrategy,
    SystemTrajectory,
    generate_nplayer_navigation_game,
    horizon,
    n_controls,
    n_states,
    solve!,
    plot_traj

using StaticArrays
using LinearAlgebra

# generate a game
T_horizon = 10.
ΔT = 0.1

"--------------------------------- Unicycle4D ---------------------------------"

x01 = @SVector [-3., 0., 0., 0.025]
x02 = @SVector [-0.1,  3.0, -pi/2, 0.05]
x03 = @SVector [0.1,  -3.0, pi/2, 0.05]
x0 = vcat(x01, x02, x03)
# goal states (goal position of other player with opposite orientation)
xg1 = @SVector [3., 0., 0., 0.]
xg2 = @SVector [-0.1, -3., 0., 0.]
xg3 = @SVector [ 0.1,  3., 0., 0.]
g = generate_nplayer_navigation_game(Unicycle4D, NPlayerUnicycleCost, T_horizon,
                                     ΔT, xg1, xg2, xg3)
zero_op = zero(SystemTrajectory, g)
h = horizon(g)
nu = n_controls(g)
nx = n_states(g)

# solve the lq game
solver = iLQSolver(g)
# - setup initial_strategy
γ_init = SizedVector{h}([AffineStrategy(@SMatrix(zeros(nu, nx)),
                                        @SVector(zeros(nu))) for k in 1:h])

# benchmark the solver
@benchmark(solve!(s.o0, s.γ0, $g, $solver, $x0),
           setup=(s=(o0=copy($zero_op), γ0=copy($γ_init))),
           samples=1000,
           evals=1) |> display

# solve the game and visualize
_ , op, _ = solve!(copy(zero_op), copy(γ_init), g, solver, x0)
plot_traj(op, g, [:red, :green, :blue]) |> display
