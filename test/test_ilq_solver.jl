using Test

using iLQGames:
    ProductSystem,
    Car5D,
    PlayerCost,
    GeneralGame,
    @S,
    quadraticize,
    linearize_discrete,
    n_states,
    n_controls,
    dynamics,
    player_costs,
    SystemTrajectory,
    lq_approximation,
    sampling_time,
    iLQSolver,
    solve,
    AffineStrategy,
    trajectory!,
    plot_systraj

using StaticArrays
using LinearAlgebra

# TODO: think about how to makethis more generic. For mocking this is okay.

struct TwoPlayerCarCost{player_id, TR<:SMatrix{2,2}, TQs<:SMatrix{5, 5},
                        TQg<:SMatrix{5,5}, TG<:SVector{5}} <: PlayerCost{10, 4}
    # the cost for control
    Rᵢ::TR
    # the state cost
    Qs::TQs
    # the cost for not being at the goal
    Qgᵢ::TQg
    # the desired goal state for this player
    xgᵢ::TG
    # the weight the asymptotic cost for being close to eachother
    qcᵢ::Float64
    # the avoidance radius
    r_avoid::Float64
    # the time after which the goal state cost is active
    t_final::Float64

end

function TwoPlayerCarCost{player_id}(R::TR, Qs::TQs, Qg::TQg, xg::TG,
                                     qc::Float64, r_avoid::Float64, t_final::Float64) where {player_id, TR, TQs, TQg, TG}
    return TwoPlayerCarCost{player_id, TR, TQs, TQg, TG}(R, Qs, Qg, xg, qc, r_avoid, t_final)
end

function (pc::TwoPlayerCarCost{player_id})(x::SVector{10}, u::SVector{4},
                                           t::Float64) where {player_id}
    # extract the states and inputs for this player
    uᵢ = u[SVector{2}(player_id == 1 ? (1:2) : (3:4))]
    xᵢ = x[SVector{5}(player_id == 1 ? (1:5) : (6:10))]
    # setup the cost: each player wan't to:
    cost = 0.
    # control:
    #   - reduce steering and acceleration/breaking effort
    # state:
    #  - avoid collisions
    #  - be close close to some target
    # control cost: only cares about own control
    cost += uᵢ' * pc.Rᵢ * uᵢ
    # cost for states (e.g. large steering)
    cost += xᵢ' * pc.Qs * xᵢ
    # goal state cost cost:
    if t > pc.t_final
        # we want to be near the goal ...
        Δxgᵢ = xᵢ - pc.xgᵢ
        cost += Δxgᵢ' * pc.Qgᵢ * Δxgᵢ
    end
    # ... but we don't want to collide (the coupling term)
    xp_other = x[SVector{2}(player_id == 1 ? (6:7) : (1:2))]
    Δxp = xp_other - xᵢ[SVector{2}(1:2)]
    # asymptotically bad to approach other player
    # cost += 1/(Δxp'*Δxp + 1) * pc.qcᵢ
    normalized_gap = Δxp' * Δxp / (2 * pc.r_avoid)
    if normalized_gap < 1.
        cost += 1//2 * (cos(normalized_gap*pi) + 1) * pc.qcᵢ
    end

    return cost
end

function generate_2player_car_game(T_horizon::Float64, ΔT::Float64)
    t_final = T_horizon - 2*ΔT

    # initial conditions:
    # x = (x, y, phi, β, v)
    ΔΘ = deg2rad(0)
    x01 = @SVector [-2., 0.3, 0., ΔΘ, 0.]
    x02 = @SVector [2., -0.3, pi, ΔΘ, 0.]
    x0 = vcat(x01, x02)
    # goal states (goal position of other player with opposite orientation)
    g1 = @SVector [2., 0., 0., 0., 0.]
    g2 = @SVector [-2., 0., pi, 0., 0.]

    # setup the dynamics
    car1 = Car5D{ΔT}(1.0)
    car2 = Car5D{ΔT}(1.0)
    dyn = ProductSystem((car1, car2))

    # cost
    # control cost
    R = @SMatrix [1. 0.; 0. 1.]
    # state cost: cost for steering angle
    Qs = SMatrix{5,5}(diagm([0, 0, 0, 20., 0]))
    # goal cost that applies only at the end of the horizon
    Qg = SMatrix{5,5}(diagm([1,1,1,0.1,1])) * 20
    # collision avoidance cost
    qc = 10.
    r_avoid = 1.

    c1 = TwoPlayerCarCost{1}(R, Qs, Qg, g1, qc, r_avoid, t_final)
    c2 = TwoPlayerCarCost{2}(R, Qs, Qg, g2, qc, r_avoid, t_final)
    costs = @SVector [c1, c2]

    # construct the game
    g = GeneralGame{((@S 1:2), (@S 3:4))}(dyn, costs)

    return g, x0
end

using Plots
pyplot()

#@testset "ilq_solver" begin
    # generate a game
    T_horizon = 10.
    ΔT = 0.1
    g, x0 = generate_2player_car_game(T_horizon, ΔT)


    # unpack for testing
    dyn = dynamics(g)
    c1, c2 = player_costs(g)

    # test quadratization of the cost
    x = @SVector rand(n_states(dyn))
    u = @SVector rand(n_controls(dyn))
    t = 0.

    # TODO: figure out why I can't use my `@inferred_with_info` macro here.
    qc1 = quadraticize(c1, x, u, t)
    qc2 = quadraticize(c2, x, u, t)
    # test linearization of the dynamics
    # TODO: currently, this put's a lot of stress on the compiler.
    linearize_discrete(dyn, x, u, t)

    # test the lq approximation:
    # generate an operating point
    h = Int(T_horizon/ΔT)
    nx = n_states(dynamics(g))
    nu = n_controls(dynamics(g))
    zero_op = zero(SystemTrajectory{h, ΔT, nx, nu})
    lqg = lq_approximation(g, zero_op)

    # the lqg approximation evaluated at zero should be approximate the true cost:

    # solve the lq game
    solver = iLQSolver()
    # TODO
    # - setup initial_strategy
    steer_init(k::Int) = sin(k/h*pi) * deg2rad(0)
    acc_init(k::Int) = -cos(k/h*pi)*0.2
    γ_init = Size(h)([AffineStrategy((@SMatrix zeros(nu, nx)),
                                     (@SVector [steer_init(k), acc_init(k),
                                                steer_init(k), acc_init(k)]))
                      for k in 1:h])
    # generate initial operating point from simulating initial strategy
    op_init = deepcopy(zero_op)
    trajectory!(op_init, dynamics(g), γ_init, zero_op, x0)
    # solve the game
    op, γ_op = @time solve(g, solver, x0, deepcopy(op_init), γ_init)

    # TODO automate posiiton coordinate extraction
    display(plot_systraj(op_init, op; xy_ids=[(1,2), (6,7)]))
#end;

# cost plots
function plot_cost(g::GeneralGame, op::SystemTrajectory, dims, i::Int=1, k::Int=1; st::Symbol=:contourf)
    lqg = lq_approximation(g, op)
    nx = n_states(dynamics(g))
    nu = n_controls(dynamics(g))
    t = k * sampling_time(dynamics(g))

    offset2vec(Δd1, Δd2) = begin
        Δx = zeros(nx)
        Δx[dims[1]] = Δd1
        if length(dims) == 2
            Δx[dims[2]] = Δd2
        end
        return SVector{nx}(Δx)
    end

    projected_cost(Δd1, Δd2=0) = begin
        Δx = offset2vec(Δd1, Δd2)
        return player_costs(g)[i](op.x[k]+Δx, op.u[k], t)
    end

    projected_cost_approx(Δd1, Δd2=0) = begin
        Δx = offset2vec(Δd1, Δd2)
        c0 = player_costs(g)[i](op.x[k], op.u[k], t)
        return player_costs(lqg)[k][i](Δx, (@SVector zeros(nu))) + c0
    end

    Δd1_range = Δd2_range = -10:0.1:10

    if length(dims) == 1
        p = plot(Δd1_range, projected_cost, label="g")
        plot!(p, Δd1_range, projected_cost_approx, label="lqg")
        return p
    elseif length(dims) == 2
        return plot(Δd1_range, Δd2_range, projected_cost, st=st)
    end

    @assert false "Can only visualize one or two dimensions"
end
