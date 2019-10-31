using StaticArrays
using BenchmarkTools

using iLQGames:
    Unicycle4D,
    NPlayerUnicycleCost,
    SystemTrajectory,
    AffineStrategy,
    LinearizationStyle,
    FeedbackLinearization,
    iLQSolver,
    GeneralGame,
    NPlayer2DDoubleIntegratorCost,
    generate_nplayer_navigation_game,
    dynamics,
    n_states,
    n_controls,
    n_players,
    player_costs,
    horizon,
    solve!,
    feedbacklin,
    player_id,
    xindex,
    uindex,
    goalcost,
    ξ_from,
    subsystems,
    transform_to_feedbacklin

# generate a game
T_horizon = 10.
ΔT = 0.1

x01 = @SVector [-3., 0., 0., 0.]
x02 = @SVector [0.,  3., 0., 0.]
x0 = vcat(x01, x02)
# goal states (goal position of other player with opposite orientation)
xg1 = @SVector [3., 0., 0., 0.]
xg2 = @SVector [0., -3., 0., 0.]
g = generate_nplayer_navigation_game(Unicycle4D, NPlayerUnicycleCost, T_horizon, ΔT,
                                     xg1, xg2)

dyn = dynamics(g)
nx = n_states(dyn)
nu = n_controls(dyn)
pcs = player_costs(g)
h = horizon(g)
zero_op = zero(SystemTrajectory{h, ΔT, nx, nu})
# solve the lq game
solver = iLQSolver(g)
# - setup initial_strategy
steer_init(k::Int) = cos(k/h*pi) * deg2rad(0)
acc_init(k::Int) = -cos(k/h*pi)*0.3
γ_init = Size(h)([AffineStrategy((@SMatrix zeros(nu, nx)),
                                 (@SVector [steer_init(k), 0.7*acc_init(k),
                                            steer_init(k), acc_init(k)])) for k in 1:h])
@info "Benchmark *without* feedback linearization:"
display(@benchmark(solve!(copy(zero_op), copy(γ_init), $g, $solver, $x0)))


steer_init(k::Int) = cos(k/h*pi) * deg2rad(0)
acc_init(k::Int) = -cos(k/h*pi)*0.3
γξ_init = Size(h)([AffineStrategy((@SMatrix zeros(nu, nx)),
                                 (@SVector [steer_init(k), 0.7*acc_init(k),
                                            steer_init(k), acc_init(k)])) for k in 1:h])

gξ, ξ0 = transform_to_feedbacklin(g, x0)
solverξ = iLQSolver(gξ)
γξ_init = Size(h)([AffineStrategy((@SMatrix zeros(nu, nx)),
                                 (@SVector [-0.05, 0.02,
                                            0.02, 0.05])) for k in 1:h])
@info "Benchmark *with* feedback linearization:"
display(@benchmark(solve!(copy(zero_op), copy(γξ_init), $gξ, $solverξ, $ξ0)))
solve!(copy(zero_op), copy(γξ_init), gξ, solverξ, ξ0)
