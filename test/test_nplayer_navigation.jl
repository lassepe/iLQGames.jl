using Test
using BenchmarkTools

using iLQGames:
    iLQGames,
    iLQSolver,
    GeneralGame,
    Unicycle4D,
    NPlayerUnicycleCost,
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
    solve!,
    ProximityCost

using iLQGames.TestUtils
using StaticArrays
using LinearAlgebra

# generate a game
T_horizon = 10.
ΔT = 0.1

"--------------------------------- Unicycle4D ---------------------------------"

x01 = @SVector [-3., 0., 0., 0.]
x02 = @SVector [0.,  3., -pi/2, 0.]
x03 = @SVector [-3.,  3., -pi/4, 0.]
x0 = vcat(x01, x02, x03)
# goal states (goal position of other player with opposite orientation)
xg1 = @SVector [3., 0., 0., 0.]
xg2 = @SVector [0., -3., -pi/2, 0.]
xg3 = @SVector [3., -3., -pi/4, 0.]
g = generate_nplayer_navigation_game(Unicycle4D, NPlayerUnicycleCost, T_horizon,
                                     ΔT, xg1, xg2, xg3;
                                     proximitycost=ProximityCost([2.0, 2.0, 2.0],
                                                                 [0., 50.0, 50.0]))
dyn = dynamics(g)
nx = n_states(dyn)
nu = n_controls(dyn)
pcs = player_costs(g)
h = horizon(g)
zero_op = zero(SystemTrajectory{h, ΔT, nx, nu})

quad_sanity_check(g)

# solve the lq game
solver = iLQSolver(g; state_regularization=5.0, control_regularization=5.0)
# - setup initial_strategy
steer_init(k::Int) = cos(k/h*pi) * deg2rad(0)
acc_init(k::Int) = -cos(k/h*pi)*0.1
γ_init = SizedVector{h}([AffineStrategy((@SMatrix zeros(nu, nx)),
                                 (@SVector [steer_init(k), 0.7*acc_init(k),
                                            steer_init(k), acc_init(k),
                                            steer_init(k), acc_init(k)])) for k in 1:h])
# generate initial operating point from simulating initial strategy
# solve the game
@benchmark(solve!(copy(zero_op), copy(γ_init), $g, $solver, $x0)) |> display
