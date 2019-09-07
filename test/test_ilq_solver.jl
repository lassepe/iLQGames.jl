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

struct TwoPlayerCarCost{player_id, TR<:SMatrix{2,2}, TQg<:SMatrix{5,5}, TG<:SVector{5}} <: PlayerCost{10, 4}
    # the cost for control
    Rᵢ::TR
    # the cost for not being at the goal
    Qgᵢ::TQg
    # the desired goal state for this player
    xgᵢ::TG
    # the weight the asymptotic cost for being close to eachother
    qcᵢ::Float64
end

function TwoPlayerCarCost{player_id}(R::TR, Qg::TQg, xg::TG, qc::Float64) where {player_id, TR, TQg, TG}
    return TwoPlayerCarCost{player_id, TR, TQg, TG}(R, Qg, xg, qc)
end

function (pc::TwoPlayerCarCost{player_id})(x::SVector{10}, u::SVector{4}, t::Float64) where {player_id}
    # setup the cost: each player wan't to:
    #
    # control:
    #   - reduce steering and acceleration/breaking effort
    # state:
    #  - avoid collisions
    #  - be close close to some target
    cost = 0
    # control cost: only cares about own control
    uᵢ = u[SVector{2}(player_id == 1 ? (1:2) : (3:4))]
    xᵢ = x[SVector{5}(player_id == 1 ? (1:5) : (6:10))]
    cost += uᵢ' * pc.Rᵢ * uᵢ

    # state cost:
    # we want to be near the goal ...
    Δxgᵢ = xᵢ - pc.xgᵢ
    cost += Δxgᵢ' * pc.Qgᵢ * Δxgᵢ
    # ... but we don't want to collide (the coupling term)
    xp_other = x[SVector{2}(player_id == 1 ? (6:7) : (1:2))]
    Δxp_other = xp_other - xᵢ[SVector{2}(1:2)]
    # asymptotically bad to approach other player
    cost += 1/(Δxp_other'*Δxp_other + 0.1) * pc.qcᵢ

    return cost
end

function generate_2player_car_game()
    ΔT = 0.1

    # initial conditions:
    # x = (x, y, phi, β, v)
    ΔΘ = deg2rad(3)
    x01 = @SVector [-2., 0., 0., ΔΘ, 0.]
    x02 = @SVector [2., 0., pi, ΔΘ, 0.]
    x0 = vcat(x01, x02)
    # goal states (goal position of other player with opposite orientation)
    g1 = @SVector [2., 0., 0., 0., 0.]
    g2 = @SVector [-2., 0., pi, 0., 0.]

    # setup the dynamics
    car1 = Car5D{ΔT}(1.0)
    car2 = Car5D{ΔT}(1.0)
    dyn = ProductSystem((car1, car2))

    # cost
    R = SMatrix{2,2}(I)*0.1
    # goal cost cares about position and orientation but not about speed and
    # steering angle
    Qg = SMatrix{5,5}(diagm([10,10,10,1,1]))
    qc = 100.

    c1 = TwoPlayerCarCost{1}(R, Qg, g1, qc)
    c2 = TwoPlayerCarCost{2}(R, Qg, g2, qc)
    costs = @SVector [c1, c2]

    # construct the game
    g = GeneralGame{((@S 1:2), (@S 3:4))}(dyn, costs)

    return g, x0
end

using Plots
pyplot()

#@testset "ilq_solver" begin
    # generate a game
    g, x0 = generate_2player_car_game()


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
    h = 30
    nx = n_states(dynamics(g))
    nu = n_controls(dynamics(g))
    ΔT = sampling_time(dynamics(g))
    zero_op = zero(SystemTrajectory{h, ΔT, nx, nu})
    lqg = lq_approximation(g, zero_op)

    # the lqg approximation evaluated at zero should be approximate the true cost:

    # solve the lq game
    solver = iLQSolver()
    # TODO
    # - setup initial_strategy
    steer_init(k::Int) = sin(k/h*pi) * deg2rad(2)
    acc_init(k::Int) = -cos(k/h*pi)*2
    γ_init = Size(h)([AffineStrategy((@SMatrix zeros(nu, nx)),
                                     (@SVector [steer_init(k), acc_init(k),
                                                steer_init(k), acc_init(k)]))
                      for k in 1:h])
    # generate initial operating point from simulating initial strategy
    op_init = deepcopy(zero_op)
    trajectory!(op_init, dynamics(g), γ_init, zero_op, x0)
    # solve the game
    op, γ_op = solve(g, solver, x0, deepcopy(op_init), γ_init)

    # TODO automate posiiton coordinate extraction
    display(plot_systraj(op_init, op; xy_ids=[(1,2), (6,7)]))
#end;

# cost plots
function plot_cost(g::GeneralGame, op::SystemTrajectory, dims, i::Int=1, k::Int=1)
    lqg = lq_approximation(g, op)
    nx = n_states(dynamics(g))
    nu = n_controls(dynamics(g))

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
        return player_costs(g)[i](op.x[k]+Δx, op.u[k], 0.)
    end

    projected_cost_approx(Δd1, Δd2=0) = begin
        Δx = offset2vec(Δd1, Δd2)
        c0 = player_costs(g)[i](op.x[k], op.u[k], 0.)
        return player_costs(lqg)[k][i](Δx, (@SVector zeros(nu))) + c0
    end

    Δd1_range = Δd2_range = -5:0.1:5

    if length(dims) == 1
        p = plot(Δd1_range, projected_cost, label="g")
        plot!(p, Δd1_range, projected_cost_approx, label="lqg")
        return p
    elseif length(dims) == 2
        return plot(Δd1_range, Δd2_range, projected_cost, st=:contourf)
    end

    @assert false "Can only visualize one or two dimensions"
end
