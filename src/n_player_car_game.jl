# TODO: tidy up. For fow, implementing both interfaces
struct NPlayerCarCost{TG<:SVector{5},TR<:SMatrix{2,2},
                      TQs<:SMatrix{5,5},TQg<:SMatrix{5,5}} <: NPlayerNavigationCost
    "the index of the player this cost applies to"
    player_id::Int
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

function NPlayerCarCost(player_id::Int, xg::TG, t_final::Float64;
                        R::TR = SMatrix{2,2}([.1 0.; 0. 1.]) * 10.,
                        Qs::TQs = SMatrix{5,5}(diagm([0, 0, 0, 0.2, 2.])) * 20.,
                        Qg::TQg = SMatrix{5,5}(diagm([1.,1.,1.,0.,0.]))*500,
                        r_avoid = 1.2,
                        gravity = 9.81,
                        des_acc_bounds = (-2*gravity, 2*gravity),
                        des_v_bounds = (-0.05, 2.),
                        des_steer_bounds = (-deg2rad(30), deg2rad(30)),
                        w = 50.) where {TXI, TUI, TG, TR, TQs, TQg}

    return NPlayerCarCost(player_id, xg, t_final, R, Qs, Qg, r_avoid,
                          des_acc_bounds, des_v_bounds, des_steer_bounds, w)
end

"---------------- Implementing the NPlayerNavigationCost interface ----------------"

player_id(pc::NPlayerCarCost) = pc.player_id
# TODO: refine to on-demand construction
inputcost(pc::NPlayerCarCost) = QuadCost(pc.R)
inputconstr(pc::NPlayerCarCost) = (SoftConstr(2, pc.des_acc_bounds..., pc.w),)
statecost(pc::NPlayerCarCost) = QuadCost(pc.Qs)
stateconstr(pc::NPlayerCarCost) = (SoftConstr(4, pc.des_steer_bounds..., pc.w),
                                   SoftConstr(5, pc.des_v_bounds..., pc.w))
proximitycost(pc::NPlayerCarCost) = ProximityCost(pc.r_avoid, pc.w)
goalcost(pc::NPlayerCarCost) = GoalCost(pc.t_final, pc.xg, pc.Qg)
