using DocStringExtensions
using LinearAlgebra
using StaticArrays

include("utils.jl")

"""
$(TYPEDEF)

A struct to represent a multi-player dynamical system.

# Parameters

- `NX`: the numer of states
- `PS`: a tuple of player symbols
- `T`:  the type of the A matrix
- `TB`: the type tuple of the b-matrices

# Fields

$(TYPEDFIELDS)
"""
struct NPlayerLinearDynamics{NP, NX, PS, T, TB<:NTuple{NP, SMatrix}}
    "The time series of state transition matrices."
    A::SMatrix{NX, NX, T}
    "A named tuple that maps a player symbol to a matrix for that player"
    B::NamedTuple{PS, TB}
end
"""
$(TYPEDEF)

A struct to represent the quadratic players costs for a singel player in a
game.

# Parameters

- `NP`: the number of players
- `NX`: the number of states
- `PS`: a tuple of player symbols
- `T`: the matrix element type (e.g. Flaot64)
- `TR`: the type tuple of R-matrices

# Fields

$(TYPEDFIELDS)
"""
struct QuadraticPlayerCost{NP, NX, PS, T, TR<:NTuple{NP, SMatrix}}
    "The qudratic state cost matrix"
    Q::SMatrix{NX, NX, T}
    "The linear state cost."
    l::SVector{NX, T}
    "A named tuple that maps a :player to the quadratic control cost matirx,
    that represents the cost that *this* player encounters for :player taking a
    certain control u"
    R::NamedTuple{PS, TR}
end

"""
$(TYPEDEF)

A struct to represent a multi-player player differential game.

# Parameters

- `NP`: the number of players
- `NX`: the number of states
- `PS` a tuple of player symbols
- `H`: the horizon of the game (number of steps, Int)
- `TD`: the type of the dynamics representation

# Fields

$(TYPEDFIELDS)
"""
struct NPlayerFiniteHorizonLQGame{NP, NX, PS, H,
                                  TD<:NPlayerLinearDynamics{NP, NX, PS},
                                  TC<:NTuple{NP, QuadraticPlayerCost{NP, NX, PS}}}
    "The linear, time varying dynamics of the system"
    linear_dynamics::SVector{H, TD}
    "The quadratic, time varying costs for each player in terms of a vector
    NamedTuples mapping :player to the corresponding cost representation."
    quadratic_cost::SVector{H, NamedTuple{PS, TC}}
end

"""
$(TYPEDSIGNATURES)

Solve a time-varying, finite horizon LQ-game to find closed-loop NASH feedback
strategies for both players.

Assumes that dynamics are given by `xₖ₊₁ = Aₖ*xₖ + ∑ᵢBₖⁱ uₖⁱ`.

# Fields

- `As`: list over time: of state dynamics matrix Aₖ

- `Bs`: list over time, over players: of control input matrix Bₖⁱ

- `Qs`: list over time, over players: of quadratic state cost Qₖⁱ

- `ls`: list over time, over players: of linear state cost lₖⁱ

- `Rs`: list over time, over `player_a` over `player_b`: of quadratic control cos
        (cost that player a sees if player b takes a certain control action)

"""

# TODO: Maybe the input to this should be a named-tuple of structs to properly
# handle the problem of type stability for different input dimensions over
# players
function solve_lq_game(As::SVector, Bs::SVector, Qs::SVector, ls::SVector,
                       Rs::SVector, val_total_u::Val{NU} = Val{2}()) where {NU}
    horizon = length(As)
    num_players = length(first(Bs))
    total_xdim = first(size(first(As)))
    # the number of controls for every player
    u_dims = SVector{num_players}([last(size(Bi)) for Bi in first(Bs)])
    u_idx_cumsum = cumsum(u_dims)
    # the index range for every player
    u_idx_range = SVector{num_players}([begin
                                            first_idx = (ii == 1 ? 1 : u_idx_cumsum[ii-1] + 1);
                                            last_idx = u_idx_cumsum[ii];
                                            SVector{u_dims[ii]}(first_idx:last_idx)
                                        end for ii in 1:num_players])
    total_udim = NU
    total_u_idx_range = SVector{total_udim}(1:total_udim)

    # initializting the optimal ocst to go representation for dynamic
    # programming
    # quadratic cost to go
    # TODO: this was the bug  beacuse the Qs were mutated! This way Z copys Q.
    Z = Vector(last(Qs))
    # linear cost to go
    ζ = Vector(last(ls))

    # TODO: figure these types out automatically. Also the dimensions might actually not be the same!
    strategies = Vector(undef, horizon)
    P_split = Vector(undef, num_players)
    α_split = Vector(undef, num_players)

    # working backwards in time to solve the dynamic program
    for kk in horizon:-1:1
        # convenience shorthands for the relevant quantities
        A = As[kk]; B = Bs[kk]; Q = Qs[kk]; l = ls[kk]; R = Rs[kk];

        # Compute Ps given previously computed Zs.
        # Refer to equation 6.17a in Basar and Olsder.
        # This will involve solving a system of matrix linear equations of the
        # form [S1s; S2s; ...] * [P1; P2; ...] = [Y1; Y2; ...].

        # Setup the S and Y matrix of the S * X = Y matrix equation
        # TODO: optimize! -- this takes too long, total_udim not known at compile time?
        # - when total_udim is known at compile time, things are sooo much faster!
        S = @SMatrix zeros(0, total_udim)
        Y = @SMatrix zeros(0, total_xdim + 1)

        for ii in 1:num_players
            BᵢZᵢ = B[ii]' * Z[ii]
            udim_ii = last(size(B[ii]))
            # the current set of rows that we construct for player ii
            S_row = @SMatrix zeros(udim_ii, 0)
            # the term for own own control cost
            for jj in 1:num_players
                # TODO: maybe think about col-major optimization here for
                # caching or windowing to avoid concatenating
                # append the column for the jth player to the current row
                S_row = hcat(S_row, (ii == jj ? R[ii][ii] + BᵢZᵢ * B[ii] : BᵢZᵢ * B[jj]))
            end
            # append the fully constructed row to the full S-Matrix
            # TODO: optimize! -- this still takes quite a bit of time.
            S = vcat(S, S_row)
            Y = vcat(Y, [(BᵢZᵢ*A) (B[ii]'*ζ[ii])])
        end

        # solve for the gains `P` and feed forward terms `α` simulatiously
        P_and_α = S \ Y
        # TODO: optimize! -- this splitting seems to cost a lot of time
        P = P_and_α[:, total_u_idx_range]
        α = P_and_α[:, end]

        for ii in 1:num_players
            P_split[ii] = P[u_idx_range[ii], :]
            α_split[ii] = α[u_idx_range[ii]]
        end

        # compute F and β as intermediate result for estimating the cost to go
        # for the next step backwards in time
        # TODO: optimize! -- the splat operator here is very slow
        # (https://github.com/JuliaArrays/StaticArrays.jl/issues/361)
        B_row_vec = static_splat(hcat, B)
        # TODO: this is weird! This should totally give the same result but
        # somehow it does not. Numerical issues?
        # Version 1
        F = A - B_row_vec * P
        β = -B_row_vec * α

        # update Z and ζ (cost to go representation for the next step backwards
        # in time)
        for ii in 1:num_players
        #    # 1. Version
        #    # TODO: optimize! -- this sum takes a lot of the time. Can we make a pure matrix expression for this?
        #    ζ[ii] = F' * (ζ[ii] + Z[ii] * β) + l[ii] + sum(P_split[jj]' * R[ii][jj] * α_split[jj] for jj in 1:num_players)
        #    # TODO: optimize! -- this sum takes a lot of the time. Can we make a pure matrix expression for this?
        #    # somehow this has to be resolved at runtime???
        #    Z[ii] = F' * Z[ii] * F + Q[ii] + sum(P_split[jj]' * R[ii][jj] * P_split[jj] for jj in 1:num_players)
            # 2. Version
            ζ[ii] = (F' * (ζ[ii] + Z[ii] * β) + l[ii])
            Z[ii] = (F' * Z[ii] * F + Q[ii])
            for jj in 1:num_players
                PjRij = P_split[jj]' * R[ii][jj]
                ζ[ii] += PjRij * α_split[jj]
                Z[ii] += PjRij * P_split[jj]
            end
        end

        # TODO: optimize! -- there must be something faster than this. Maybe
        # look at how they do it in `DynamicalSystems.jl` and/or `DifferentialEquations.jl`
        strategies[kk] = (P_split, α_split)
    end

    return strategies
end

# TODO: remove -- test code:
Q1 = @SMatrix [1. 0.; 0. 1.]
l1 = @SVector [1., 2.]
R1 = (P1=(@SMatrix [1.0 0.0; 0.0 1.0]), P2=(@SMatrix [1.0]))
qpc1 = QuadraticPlayerCost(Q1, l1, R1)

Q2 = -Q1
l2 = -l1
R2 = R1
qpc2 = QuadraticPlayerCost(Q2, l2, R2)

# A simple dynamical system:
A = @SMatrix [1.0 1.0; 0.0 1.0]
B1 = @SMatrix [1.0 0.0; 0.0 1.0]
B2 = @SMatrix [1.0; 0.0]
npld = NPlayerLinearDynamics(A, (P1=B1, P2=B2))

# constructing the game
const H = 100
ltv_dynamcis = SVector{H}(repeat([npld], H))
qtv_cost = SVector{H}(repeat([(P1=qpc1, P2=qpc2)], H))

lqGame = NPlayerFiniteHorizonLQGame(ltv_dynamcis, qtv_cost)
