using Test
using BenchmarkTools

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
    iLQSolver,
    solve,
    AffineStrategy,
    QuadraticPlayerCost,
    softconstr,
    softconstr_quad!,
    proximitycost,
    proximitycost_quad!,
    goalstatecost,
    goalstatecost_quad!,
    statecost,
    statecost_quad!,
    inputcost,
    inputcost_quad!

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
    R::TR = SMatrix{2,2}([.1 0.; 0. 1.]) * 10.
    # the state cost
    Qs::TQs = SMatrix{5,5}(diagm([0, 0, 0, 0.1, 2.])) * 20.
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
    inputcost_quad!(R, pc.R, ui)
    # soft constraints on the control
    # - acceleration constraint
    softconstr_quad!(R, u, pc.des_acc_bounds..., pc.w, ui[2])

    # the quadratic part of the state cost
    statecost_quad!(Q, l, pc.Qs, x[xi], xi)
    # soft constraints on the state
    # - steering angle
    softconstr_quad!(Q, l, x, pc.des_steer_bounds..., pc.w, xi[4])
    # - speed
    softconstr_quad!(Q, l, x, pc.des_v_bounds..., pc.w, xi[5])
    # - proximity
    proximitycost_quad!(Q, l, x, pc.r_avoid, pc.w, 1, 2, 6, 7)

    # the goal cost
    goalstatecost_quad!(Q, l, pc.Qg, pc.xg, x[xi], xi, t, pc.t_final)

    return QuadraticPlayerCost(SMatrix(Q), SVector(l), SMatrix(R))
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
    cost += inputcost(pc.R, uᵢ)
    # running cost for states (e.g. large steering)
    cost += statecost(pc.Qs, xᵢ)

    # acceleration constraints
    cost += softconstr(uᵢ[2], pc.des_acc_bounds..., pc.w)
    # steering angle constraint
    cost += softconstr(xᵢ[4], pc.des_steer_bounds..., pc.w)
    # speed constraints
    cost += softconstr(xᵢ[5], pc.des_v_bounds..., pc.w)

    # proximity constraint
    xp_other = x[SVector{2}(pc.player_id == 1 ? (6:7) : (1:2))]
    cost += proximitycost(xᵢ[(@S 1:2)], xp_other, pc.r_avoid, pc.w)

    # goal state cost cost:
    cost += goalstatecost(pc.Qg, pc.xg, xᵢ, t, pc.t_final)

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

@testset "LQ Approximation Sanity Check." begin
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
        linearize_discrete(dyn, x, u, t)
    end
end;


# test the lq approximation:
# generate an operating point
h = Int(T_horizon/ΔT)
zero_op = zero(SystemTrajectory{h, ΔT, nx, nu})
lqg = lq_approximation(g, zero_op)

# the lqg approximation evaluated at zero should be approximate the true cost:

# solve the lq game
solver = iLQSolver()
# - setup initial_strategy
steer_init(k::Int) = cos(k/h*pi) * deg2rad(0)
acc_init(k::Int) = -cos(k/h*pi)*0.3
γ_init = Size(h)([AffineStrategy((@SMatrix zeros(nu, nx)),

                                 (@SVector [steer_init(k), 0.7*acc_init(k),
                                            steer_init(k), acc_init(k)])) for k in 1:h])
# generate initial operating point from simulating initial strategy
# solve the game
display(@benchmark solve($g, $solver, $x0, $zero(zero_op), $γ_init))
