using Test

using iLQGames:
    ProductSystem,
    Car5D,
    PlayerCost,
    GeneralGame,
    @S

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
    xp_other = x[SVector{2}(player_id == 2 ? (1:2) : (6:7))]
    Δxp_other = xp_other - xᵢ[SVector{2}(1:2)]
    # asymptotically bad to approach other player
    cost += 1/(norm(Δxp_other) + 0.01) * Qcᵢ

    return cost
end

function generate_2player_car_game()
    ΔT = 0.1
    H = 10.0
    N_STEPS = Int(H / ΔT)

    # initial conditions:
    # x = (x, y, phi, β, v)
    x01 = @SVector [-5., 0., 0., 0., 0.]
    x02 = @SVector [5., 0., pi, 0., 0.]
    # goal states (goal position of other player with opposite orientation)
    g1 = @SVector [5., 0., 0., 0., 0.]
    g2 = @SVector [5., 0., pi, 0., 0.]

    # setup the dynamics
    car1 = Car5D(1.0)
    car2 = Car5D(1.0)
    dyn = ProductSystem((car1, car2))

    # cost
    R = SMatrix{2,2}(I)
    # goal cost cares about position and orientation but not about speed and
    # steering angle
    Qg = SMatrix{5,5}(diagm([1,1,1,0,0]))
    qc = 1.

    c1 = TwoPlayerCarCost{1}(R, Qg, g1, qc)
    c2 = TwoPlayerCarCost{2}(R, Qg, g2, qc)
    costs = @SVector [c1, c2]

    # construct the game
    g = GeneralGame{((@S 1:2), (@S 3:4))}(dyn, costs)
end
