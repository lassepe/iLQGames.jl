using Test

using iLQGames:
    iLQGames,
    ProductSystem,
    Car5D,
    PlayerCost,
    GeneralGame,
    @S,
    quadraticize,
    _quadraticize_ad,
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
    next_x,
    animate_plot,
    uindex,
    @animated,
    plot_traj,
    QuadraticPlayerCost

import iLQGames:
    quadraticize

using StaticArrays
using LinearAlgebra
using Parameters

@with_kw struct TwoPlayerCarCost{TR<:SMatrix{2,2}, TQs<:SMatrix{5, 5},
                                 TQg<:SMatrix{5,5}, TG<:SVector{5}} <:
    PlayerCost{10, 4}
    # an unique identifier for this player
    player_id::Int
    # the desired goal state for this player
    xg::TG
    # the time after which the goal state cost is active
    t_final::Float64
    # the cost for control
    Rᵢ::TR = SMatrix{2,2}([.1 0.; 0. 1.]) * 5.
    # the state cost
    Qs::TQs = SMatrix{5,5}(diagm([0, 0, 0, 0.1, 2.])) * 10.
    # the cost for not being at the goal
    Qg::TQg = SMatrix{5,5}(diagm([1.,1.,1.,0.,0.]))*500
    # the avoidance radius
    r_avoid::Float64 = 1.
    # soft constraints
    # the gravidty constant
    gravity::Float64 = 9.81
    # bounds on the acceleration input
    des_acc_bounds::Tuple{Float64, Float64} = (-2*gravity, 2*gravity)
    # bounds on the velocity state
    des_v_bounds::Tuple{Float64, Float64} = (-0.05, 2.)
    # bounds on the steering state
    des_steer_bounds::Tuple{Float64, Float64} = (-deg2rad(30), deg2rad(30))
    # weight of the soft constraints
    w::Float64 = 500.
end

function iLQGames.quadraticize(pc::TwoPlayerCarCost, x::SVector{10},
                               u::SVector{4}, t::AbstractFloat)

    xi = pc.player_id == 1 ? (@SVector [1,2,3,4,5]) : (@SVector [6,7,8,9,10])
    ui = pc.player_id == 1 ? (@SVector [1,2]) : (@SVector [3,4])

    l = @MVector zeros(10)
    Q = @MMatrix zeros(10, 10)
    R = @MMatrix zeros(4, 4)

    # the quadratic part of the control cost
    R[ui, ui] += 2pc.Rᵢ
    # soft constraints on the control
    # - steering rate constraint
    # - acceleration constraint
    softconstr_quad!(R, u, pc.des_acc_bounds..., pc.w, ui[2])

    # the quadratic part of the state cost
    Q[xi, xi] += 2pc.Qs
    l[xi] += 2pc.Qs*x[xi]
    # soft constraints on the state
    # - steering angle
    softconstr_quad!(Q, l, x, pc.des_steer_bounds..., pc.w, xi[4])
    # - speed
    softconstr_quad!(Q, l, x, pc.des_v_bounds..., pc.w, xi[5])
    # - proximity
    proximity_quad!(Q, l, x, pc.r_avoid, pc.w, 1, 2, 6, 7)

    # the goal cost
    if t > pc.t_final
        Qg = 2pc.Qg
        Q[xi, xi] += Qg
        l[xi] += Qg*(x[xi]-pc.xg)
    end

    return QuadraticPlayerCost(SMatrix(Q), SVector(l), SMatrix(R))
end

@inline function proximity_quad!(Q::MMatrix, l::MVector, x::SVector,
                                 r_avoid::Real, w::Real, x1::Int, y1::Int,
                                 x2::Int, y2::Int)
    Δx = x[x1] - x[x2]
    Δy = x[y1] - x[y2]
    Δsq = Δx^2 + Δy^2
    if Δsq < r_avoid
        # cost model: w*(Δsq - min)^2
        δx = 4*w*Δx*(Δsq - r_avoid)
        δy = 4*w*Δy*(Δsq - r_avoid)
        l[x1] += δx
        l[y1] += δy
        l[x2] -= δx
        l[y2] -= δy

        δxδx = 4w*(Δsq - r_avoid) + 8w*Δx^2
        δyδy = 4w*(Δsq - r_avoid) + 8w*Δy^2
        δxδy = 8w*Δx*Δy

        Q[x1,x1] += δxδx
        Q[x1,x2] -= δxδx; Q[x2,x1] -= δxδx
        Q[x2,x2] += δxδx

        Q[y1,y1] += δyδy
        Q[y1,y2] -= δyδy; Q[y2,y1] -= δyδy
        Q[y2,y2] += δyδy

        Q[x1,y1] += δxδy; Q[y1,x1] += δxδy
        Q[x1,y2] -= δxδy; Q[y2,x1] -= δxδy
        Q[y1,x2] -= δxδy; Q[x2,y1] -= δxδy
        Q[x2,y2] += δxδy; Q[y2,x2] += δxδy
    end
end

@inline function softconstr_quad!(Q::MMatrix, l::MVector, x::SVector,
                                  min::Real, max::Real, w::Real, idx::Int)
    @assert min < max
    if x[idx] < min
        Q[idx, idx] += 2w
        l[idx] += 2w*(x[idx] - min)
    elseif x[idx] > max
        Q[idx, idx] += 2w
        l[idx] += 2w*(x[idx] - max)
    end
end

@inline function softconstr_quad!(Q::MMatrix, x::SVector, min::Real, max::Real,
                                  w::Real, idx::Int)
    @assert min < max
    if !(min < x[idx] < max)
        Q[idx, idx] += 2w
    end
end

@inline function softconstr(val::Real, min::Real, max::Real, w::Real)
    @assert min < max
    gap = val < min ? val - min : val > max ? val - max : zero(w)
    return w*gap^2
end

function (pc::TwoPlayerCarCost)(x::SVector{10}, u::SVector{4}, t::Float64)
    # extract the states and inputs for this player
    uᵢ = u[SVector{2}(pc.player_id == 1 ? (1:2) : (3:4))]
    xᵢ = x[SVector{5}(pc.player_id == 1 ? (1:5) : (6:10))]
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

    # acceleration constraints
    cost += softconstr(uᵢ[2], pc.des_acc_bounds..., pc.w)
    # steering angle constraint
    cost += softconstr(xᵢ[4], pc.des_steer_bounds..., pc.w)
    # speed constraints
    cost += softconstr(xᵢ[5], pc.des_v_bounds..., pc.w)

    # proximity constraint
    xp_other = x[SVector{2}(pc.player_id == 1 ? (6:7) : (1:2))]
    Δxp = xp_other - xᵢ[SVector{2}(1:2)]
    cost += softconstr(Δxp'*Δxp, pc.r_avoid, Inf, pc.w)

    # goal state cost cost:
    if t > pc.t_final
        # we want to be near the goal ...
        Δxg = xᵢ - pc.xg
        cost += Δxg' * pc.Qg * Δxg
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

    c1 = TwoPlayerCarCost(player_id=1, xg=g1, t_final=t_final)
    c2 = TwoPlayerCarCost(player_id=2, xg=g2, t_final=t_final)
    costs = @SVector [c1, c2]

    # construct the game
    g = GeneralGame{((@S 1:2), (@S 3:4))}(dyn, costs)

    return g, x0
end

#@testset "ilq_solver" begin
    # generate a game
    T_horizon = 10.
    ΔT = 0.1
    g, x0 = generate_2player_car_game(T_horizon, ΔT)


    # unpack for testing
    dyn = dynamics(g)
    nx = n_states(dyn)
    nu = n_controls(dyn)
    c1, c2 = player_costs(g)

    # test quadratization of the cost and quadratization of cost
    x = @SVector zeros(nx)
    u = @SVector zeros(nu)
    t = 0.
    for i in 1:100
        global x = SVector{nx, Float64}(randn(nx))
        global u = SVector{nu, Float64}(randn(nu))
        global t = T_horizon

        for c in (c1, c2)
            qc = [quad(c, x, u, t) for quad in (quadraticize, _quadraticize_ad)]
            @test isapprox(qc[1].Q, qc[2].Q)
            @test isapprox(qc[1].l, qc[2].l)
            @test isapprox(qc[1].R, qc[2].R)
        end

        # test linearization of the dynamics
        # TODO: currently, this put's a lot of stress on the compiler.
        linearize_discrete(dyn, x, u, t)

    end


    # test the lq approximation:
    # generate an operating point
    h = Int(T_horizon/ΔT)
    zero_op = zero(SystemTrajectory{h, ΔT, nx, nu})
    lqg = lq_approximation(g, zero_op)

    # the lqg approximation evaluated at zero should be approximate the true cost:

    # solve the lq game
    solver = iLQSolver()
    # TODO
    # - setup initial_strategy
    steer_init(k::Int) = cos(k/h*pi) * deg2rad(0)
    acc_init(k::Int) = -cos(k/h*pi)*0.3
    γ_init = Size(h)([AffineStrategy((@SMatrix zeros(nu, nx)),

                                     (@SVector [steer_init(k), 0.7*acc_init(k),
                                                steer_init(k), acc_init(k)])) for k in 1:h])
    # generate initial operating point from simulating initial strategy
    op_init = deepcopy(zero_op)
    trajectory!(op_init, dynamics(g), γ_init, zero_op, x0)
    # solve the game
    op, γ_op = @time solve(g, solver, x0, zero_op, γ_init)

    # TODO automate posiiton coordinate extraction
    op_init
    print("""
          Top -- init: $(cost(g, op_init))
          Bottom -- normal: $(cost(g, op))
          """)

   using Plots
   pyplot()
   default(size=(600, 300))
   display(plot(plot_traj(op_init, ((@S 1:2), (@S 6:7)), uindex(g)),
                plot_traj(op, ((@S 1:2), (@S 6:7)), uindex(g)),
                layout=(2, 1)))
