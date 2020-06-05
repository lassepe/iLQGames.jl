using Test
using BenchmarkTools

using iLQGames:
    iLQGames,
    Unicycle4D,
    NPlayerUnicycleCost,
    quadraticize,
    _quadraticize_ad,
    linearize_discrete,
    n_states,
    n_controls,
    dynamics,
    player_costs,
    SystemTrajectory,
    lq_approximation,
    iLQSolver,
    solve!,
    AffineStrategy,
    generate_nplayer_navigation_game,
    samplingtime,
    horizon

using iLQGames.TestUtils

using StaticArrays
using LinearAlgebra


# generate a game
T_horizon = 10.
ΔT = 0.1
# initial conditions:
# x = (x, y, phi, β, v)
x01 = @SVector [-3., 0., 0., 0.]
x02 = @SVector [0.,  3., -pi/2, 0.]
x0 = vcat(x01, x02)
# goal states (goal position of other player with opposite orientation)
xg1 = @SVector [3., 0., 0., 0.]
xg2 = @SVector [0., -3., -pi/2, 0.]
# generate game
g = generate_nplayer_navigation_game(Unicycle4D, NPlayerUnicycleCost, T_horizon, ΔT, xg1, xg2)

# unpack for testing
dyn = dynamics(g)
nx = n_states(dyn)
nu = n_controls(dyn)
c1, c2 = player_costs(g)

# test quadraticization of the cost and quadratization of cost
x = @SVector zeros(nx)
u = @SVector zeros(nu)
t = 0.
# solve the lq game
solver = iLQSolver(g)

quad_sanity_check(g)

# test the lq approximation:
# generate an operating point
h = Int(T_horizon/ΔT)
zero_op = zero(SystemTrajectory{h, ΔT, nx, nu})
lqg = lq_approximation(solver, g, zero_op)

# the lqg approximation evaluated at zero should be approximate the true cost:

# - setup initial_strategy
steer_init(k::Int) = cos(k/h*pi) * deg2rad(0)
acc_init(k::Int) = -cos(k/h*pi)*0.3
γ_init = SizedVector{h}([AffineStrategy((@SMatrix zeros(nu, nx)),

                                 (@SVector [steer_init(k), 0.7*acc_init(k),
                                            steer_init(k), acc_init(k)])) for k in 1:h])
# generate initial operating point from simulating initial strategy
# solve the game
@benchmark(solve!(copy(zero_op), copy(γ_init), $g, $solver, $x0)) |> display

_, traj, _ = solve!(copy(zero_op), copy(γ_init), g, solver, x0)
@testset "solution trajectory sanity checks" begin
    @test horizon(traj) == h
    @test samplingtime(traj) == ΔT
end
