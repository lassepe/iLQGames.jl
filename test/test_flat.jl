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
    subsystems

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
                                 (@SVector zeros(nu))) for k in 1:h])
@info "Solve *without* feedback linearization:"
# display(@benchmark(solve!(copy(zero_op), copy(γ_init), $g, $solver, $x0)))

# TODO: FIXME -- for now manually transformed to linearized form
function transform_to_feedbacklin(g::GeneralGame, x0)
    @assert LinearizationStyle(dynamics(g)) isa FeedbackLinearization
    # transform the dynamics
    lin_dyn = feedbacklin(dynamics(g))
    # approximate the cost by a cost in ξ coordinates
    # TODO: do more generically
    ξ_cost = map(enumerate(player_costs(g))) do (i, c)
        ξg = ξ_from(subsystems(dynamics(g))[i], goalcost(c).xg)
        return NPlayer2DDoubleIntegratorCost(player_id(c), xindex(c), uindex(c), ξg,
                                             goalcost(c).t_active)
    end |> SVector{n_players(g)}

    return GeneralGame{uindex(g), horizon(g)}(lin_dyn, ξ_cost), ξ_from(dynamics(g),
                                                                       x0)
end

gξ, ξ0 = transform_to_feedbacklin(g, x0)
solverξ = iLQSolver(gξ)
# TODO: also transform strategies and operating point

display(@benchmark(solve!(copy(zero_op), copy(γ_init), $gξ, $solverξ, $ξ0)))
