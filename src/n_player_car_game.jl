# TODO: tidy up. For fow, implementing both interfaces
struct NPlayerCarCost{nx,nu,TXI<:NTuple,TUI<:NTuple,TG<:SVector{5},TR<:SMatrix{2,2},
                      TQs<:SMatrix{5,5},TQg<:SMatrix{5,5}} <: NPlayerNavigationCost{nx,nu}
    "the index of the player this cost applies to"
    player_id::Int
    "the index partitions of the state vector"
    xids::TXI
    "the index partitions of the input vector"
    uids::TUI
    "the desired goal state for this player"
    xg::TG
    "the time after which the goal state cost is active"
    t_final::Float64
    "the cost for control"
    R::TR
    "the state cost"
    Qs::TQs
    "the cost for not being at the goal"
    Qg::TQg
    "the avoidance radius"
    r_avoid::Float64
    # soft constraints
    "bounds on the acceleration input"
    des_acc_bounds::Tuple{Float64, Float64}
    "bounds on the velocity state"
    des_v_bounds::Tuple{Float64, Float64}
    "bounds on the steering state"
    des_steer_bounds::Tuple{Float64, Float64}
    "weight of the soft constraints"
    w::Float64
end

function NPlayerCarCost(player_id::Int, xids::TXI, uids::TUI, xg::TG, t_final::Float64;
                        R::TR = SMatrix{2,2}([.1 0.; 0. 1.]) * 10.,
                        Qs::TQs = SMatrix{5,5}(diagm([0, 0, 0, 0.2, 2.])) * 20.,
                        Qg::TQg = SMatrix{5,5}(diagm([1.,1.,1.,0.,0.]))*500,
                        r_avoid = 1.2,
                        gravity = 9.81,
                        des_acc_bounds = (-2*gravity, 2*gravity),
                        des_v_bounds = (-0.05, 2.),
                        des_steer_bounds = (-deg2rad(30), deg2rad(30)),
                        w = 50.) where {TXI, TUI, TG, TR, TQs, TQg}
    nx = sum(length.(xids))
    nu = sum(length.(uids))
    return NPlayerCarCost{nx,nu,TXI,TUI,TG,TR,TQs,TQg}(player_id, xids, uids, xg,
                                                       t_final, R, Qs, Qg,
                                                       r_avoid,
                                                       des_acc_bounds,
                                                       des_v_bounds,
                                                       des_steer_bounds, w)
end

"---------------- Implementing the NPlayerNavigationCost interface ----------------"

player_id(pc::NPlayerCarCost) = pc.player_id
# TODO: refine to on-demand construction
inputcost(pc::NPlayerCarCost) = InputCost(pc.R)
inputconstr(pc::NPlayerCarCost) = (SoftConstr(2, pc.des_acc_bounds..., pc.w),)
statecost(pc::NPlayerCarCost) = StateCost(pc.Qs)
stateconstr(pc::NPlayerCarCost) = (SoftConstr(4, pc.des_steer_bounds..., pc.w),
                                   SoftConstr(5, pc.des_v_bounds..., pc.w))
proximitycost(pc::NPlayerCarCost) = ProximityCost(pc.r_avoid, pc.w)
goalcost(pc::NPlayerCarCost) = GoalCost(pc.t_final, pc.xg, pc.Qg)
xindex(pc::NPlayerCarCost) = pc.xids
uindex(pc::NPlayerCarCost) = pc.uids

"---------------------- Legacy functions for sanity checking ----------------------"

function _legacy_cost(pc::NPlayerCarCost, x::SVector, u::SVector, t::Float64)
    # extract the states and inputs for this player
    xᵢ = x[xindex(pc)[pc.player_id]]
    uᵢ = u[uindex(pc)[pc.player_id]]
    # setup the cost: each player wan't to:
    cost = 0.

    # control cost: only cares about own control
    cost += inputcost(pc.R, uᵢ)
    # acceleration constraints
    cost += softconstr(uᵢ[2], pc.des_acc_bounds..., pc.w)

    # running cost for states (e.g. large steering)
    cost += statecost(pc.Qs, xᵢ)
    # steering angle constraint
    cost += softconstr(xᵢ[4], pc.des_steer_bounds..., pc.w)
    # speed constraints
    cost += softconstr(xᵢ[5], pc.des_v_bounds..., pc.w)

    # proximity constraint
    xy_ego = xᵢ[@S(1:2)]
    for (j, xj) in enumerate(xindex(pc))
        j != pc.player_id || continue
        xy_other = x[xj[@S(1:2)]]
        cost += proximitycost(xy_ego, xy_other, pc.r_avoid, pc.w)
    end

    # goal state cost cost:
    cost += goalstatecost(pc.Qg, pc.xg, xᵢ, t, pc.t_final)

    return cost
end
