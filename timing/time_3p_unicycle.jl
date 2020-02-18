using Revise
using BenchmarkTools

import iLQGames: force_ad_use

using iLQGames:
    iLQSolver,
    ControlSystem,
    PlayerCost,
    Unicycle4D,
    NPlayerUnicycleCost,
    AffineStrategy,
    SystemTrajectory,
    generate_nplayer_navigation_game,
    horizon,
    n_controls,
    n_states,
    solve!,
    plot_traj,
    transform_to_feedbacklin

using StaticArrays
using LinearAlgebra

# generate a game
T_horizon = 10.
ΔT = 0.1

macro benchmark_solver(g, solver, x0)
    return quote
        sleep(3)
        @benchmark(solve!(s.o0, s.γ0, $g, $solver, $x0),
                   setup=(s=(o0=copy($zero_op), γ0=copy($γ_init))),
                   samples=100,
                   evals=1,
                   seconds=30) |> display
        _ , op, _ = solve!(copy(zero_op), copy(γ_init), $g, $solver, $x0)
        plot_traj(op, $g, [:red, :green, :blue]) |> display
    end
end

"--------------------------------- Nonlinear Unicycle4D ---------------------------"

x01 = @SVector [-3., 0., 0., 0.05]
x02 = @SVector [-0.1,  3.0, -pi/2, 0.1]
x03 = @SVector [0.1,  -3.0, pi/2, 0.1]
x0 = vcat(x01, x02, x03)
# goal states (goal position of other player with opposite orientation)
xg1 = @SVector [3., 0., 0., 0.05]
xg2 = @SVector [-0.1, -3., 0., 0.1]
xg3 = @SVector [ 0.1,  3., 0., 0.1]
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

println("Nonlinear Unicycle4D")
println("Manual Differentiation")
force_ad_use(::ControlSystem) = false
force_ad_use(::PlayerCost) = false
@benchmark_solver(g, solver, x0)

println("Automatic Differentiation")
force_ad_use(::ControlSystem) = true
force_ad_use(::PlayerCost) = true
@benchmark_solver(g, solver, x0)

"------------------------------ Flat Unicycle4D -----------------------------------"
gξ, ξ0 = transform_to_feedbacklin(g, x0)
solverξ = iLQSolver(gξ)

println("=====================")
println("Feedback linearized Unicycle4D")
println("Manual Differentiation")
force_ad_use(::ControlSystem) = false
force_ad_use(::PlayerCost) = false
@benchmark_solver(gξ, solverξ, ξ0)

println("Automatic Differentiation")
force_ad_use(::ControlSystem) = true
force_ad_use(::PlayerCost) = true
@benchmark_solver(gξ, solverξ, ξ0)
