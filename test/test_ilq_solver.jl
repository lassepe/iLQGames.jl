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
    plot_traj,
    cost,
    next_x

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
    # running cost for states (e.g. large steering)
    cost += xᵢ' * pc.Qs * xᵢ

    @inline function soft_constraint(val::Real, min::Real, max::Real, w::Real)
        @assert min < max
        gap = 0.
        if val < min
            gap = min - val
        elseif val > max
            gap = val - max
        end
        return gap*gap*w
    end

    # soft constraints
    # maximum acceleration
    gravity = 9.81
    des_acc_min = -4*gravity
    des_acc_max = -2*gravity
    des_v_min = -0.05
    des_v_max = 8.
    des_steer_min= -deg2rad(30)
    des_steer_max = -des_steer_min
    w = 100.

    # steering angle constraint
    cost += soft_constraint(uᵢ[1], des_steer_min, des_steer_max, w)
    # acceleration constraints
    cost += soft_constraint(uᵢ[2], des_acc_min, des_acc_max, w)

    # speed constraints
    cost += soft_constraint(xᵢ[5], des_v_min, des_v_max, w)

    # proxiimity constraint
    xp_other = x[SVector{2}(player_id == 1 ? (6:7) : (1:2))]
    Δxp = xp_other - xᵢ[SVector{2}(1:2)]
    cost += soft_constraint(Δxp'*Δxp, pc.r_avoid, Inf, w)

    # goal state cost cost:
    if t > pc.t_final
        # we want to be near the goal ...
        Δxgᵢ = xᵢ - pc.xgᵢ
        cost += Δxgᵢ' * pc.Qgᵢ * Δxgᵢ
    end

    return cost
end

function generate_2player_car_game(T_horizon::Float64, ΔT::Float64)
    t_final = T_horizon - 1.5*ΔT

    # initial conditions:
    # x = (x, y, phi, β, v)
    x01 = @SVector [-3., 0., 0., 0., 0.]
    x02 = @SVector [0.,  3., -pi/2, 0., 0.]
    x0 = vcat(x01, x02)
    # goal states (goal position of other player with opposite orientation)
    g1 = @SVector [3., 0., 0., 0., 0.]
    g2 = @SVector [0., -3., -pi/2, 0., 0.]

    # setup the dynamics
    car1 = Car5D{ΔT}(1.0)
    car2 = Car5D{ΔT}(1.0)
    dyn = ProductSystem((car1, car2))

    # cost
    # control cost
    R = SMatrix{2,2}([1. 0.; 0. 1.]) * 0.1
    # state cost: cost for steering angle
    Qs = SMatrix{5,5}(diagm([0, 0, 0, 10., 1.])) * 0.1
    # goal cost that applies only at the end of the horizon
    Qg = SMatrix{5,5}(diagm([1.,1.,1.,0.,0.]))*100
    # collision avoidance cost
    qc = 0.
    r_avoid = 1.

    c1 = TwoPlayerCarCost{1}(R, Qs, Qg, g1, qc, r_avoid, t_final)
    c2 = TwoPlayerCarCost{2}(R, Qs, Qg, g2, qc, r_avoid, t_final)
    costs = @SVector [c1, c2]

    # construct the game
    g = GeneralGame{((@S 1:2), (@S 3:4))}(dyn, costs)

    return g, x0
end

function flip_steering(op::SystemTrajectory)
    op_flip = zero(typeof(op))
    xₖ = op.x[1]
    for k in eachindex(op_flip.x)
        op_flip.x[k] = xₖ
        op_flip.u[k] = SMatrix{4,4}(diagm([-1, 1, -1, 1]))*op.u[k]
        xₖ = next_x(dynamics(g), xₖ, op_flip.u[k], k)
    end
    return op_flip
end


using Plots
pyplot()

#@testset "ilq_solver" begin
    # generate a game
    T_horizon = 10.
    ΔT = 0.05
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
    steer_init(k::Int) = cos(k/h*pi) * deg2rad(0)
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
    op_init
    print("""
          Top -- init: $(cost(g, op_init))
          Bottom -- normal: $(cost(g, op))
          """)
    display(plot_traj(op_init, op; xy_ids=[(1,2), (6,7)], uids=uindex(g)))
#end;

# cost plots
function plot_cost(g::GeneralGame, op::SystemTrajectory, dims, i::Int=1, k::Int=1; st::Symbol=:contour)
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

function animate_cost(g::GeneralGame, op::SystemTrajectory,
                      k_range::Union{UnitRange, Nothing}=nothing, dims=(1:2),
                      i::Int=1; st::Symbol=:contour, frame_sampling::Int=2,
                      fps=10,
                      filename::String="$(@__DIR__)/../debug_out/test.gif")
    k_range = isnothing(k_range) ? (1:length(op.x)) : k_range
    anim = @animate for k in k_range
        plot_cost(g, op, dims, i, k, st=st)
    end every frame_sampling
    gif(anim, filename, fps=fps)
end
