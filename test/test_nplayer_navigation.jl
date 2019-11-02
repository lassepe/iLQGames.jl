using Test
using BenchmarkTools

using iLQGames:
    iLQGames,
    iLQSolver,
    GeneralGame,
    Unicycle4D,
    Car5D,
    NPlayerUnicycleCost,
    NPlayerCarCost,
    AffineStrategy,
    SystemTrajectory,
    dynamics,
    n_states,
    n_controls,
    horizon,
    player_costs,
    quadraticize,
    _quadraticize_ad,
    generate_nplayer_navigation_game,
    solve!

using iLQGames.TestUtils

using StaticArrays
using LinearAlgebra

# generate a game
T_horizon = 10.
ΔT = 0.1

"-------------------------------------- Car5D -------------------------------------"

# initial conditions:
# x = (x, y, phi, β, v)
x01 = @SVector [-3., 0., 0., 0., 0.]
x02 = @SVector [0.,  3., -pi/2, 0., 0.]
x0 = vcat(x01, x02)
# goal states (goal position of other player with opposite orientation)
xg1 = @SVector [3., 0., 0., 0., 0.]
xg2 = @SVector [0., -3., -pi/2, 0., 0.]
g = generate_nplayer_navigation_game(Car5D, NPlayerCarCost, T_horizon,
                                     ΔT, xg1, xg2)
# generate game

# unpack for testing
dyn = dynamics(g)
nx = n_states(dyn)
nu = n_controls(dyn)
pcs = player_costs(g)

# test quadratization of the cost and quadratization of cost
x = @SVector zeros(nx)
u = @SVector zeros(nu)
t = 0.

@testset "Scalar comparison old new." begin
    for i in 1:100
        x = SVector{nx, Float64}(randn(nx))
        u = SVector{nu, Float64}(randn(nu))
        t = T_horizon

        for pc in pcs
            c_old = iLQGames._legacy_cost(pc, x, u, t)
            c_new = pc(g, x, u, t)
            @test isapprox(c_old, c_new)
        end
    end
end;

quad_sanity_check(g)

"--------------------------------- Unicycle4D ---------------------------------"

x01 = @SVector [-3., 0., 0., 0.]
x02 = @SVector [0.,  3., 0., 0.]
x0 = vcat(x01, x02)
# goal states (goal position of other player with opposite orientation)
xg1 = @SVector [3., 0., 0., 0.]
xg2 = @SVector [0., -3., 0., 0.]
g = generate_nplayer_navigation_game(Unicycle4D, NPlayerUnicycleCost, T_horizon,
                                     ΔT, xg1, xg2)
dyn = dynamics(g)
nx = n_states(dyn)
nu = n_controls(dyn)
pcs = player_costs(g)
h = horizon(g)
zero_op = zero(SystemTrajectory{h, ΔT, nx, nu})

quad_sanity_check(g)

# solve the lq game
solver = iLQSolver(g)
# - setup initial_strategy
steer_init(k::Int) = cos(k/h*pi) * deg2rad(0)
acc_init(k::Int) = -cos(k/h*pi)*0.3
γ_init = Size(h)([AffineStrategy((@SMatrix zeros(nu, nx)),
                                 (@SVector [steer_init(k), 0.7*acc_init(k),
                                            steer_init(k), acc_init(k)])) for k in 1:h])
# generate initial operating point from simulating initial strategy
# solve the game
display(@benchmark(solve!(copy(zero_op), copy(γ_init), $g, $solver, $x0)))
