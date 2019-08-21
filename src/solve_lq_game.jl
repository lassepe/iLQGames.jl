using DocStringExtensions
using LinearAlgebra
using StaticArrays

include("utils.jl")

"""
$(TYPEDEF)

A struct to represent a multi-player dynamical system.

# Parameters

- `nx`: the numer of states
- `np`: the number of players
- `ps`: a tuple of player symbols
- `T`:  the type of the A matrix
- `TB`: the type tuple of the b-matrices

# Fields

$(TYPEDFIELDS)
"""
struct NPlayerLinearDynamics{nx, np, ps, T, TB<:NTuple{np, SMatrix}}
    "The time series of state transition matrices."
    A::SMatrix{nx, nx, T}
    "A named tuple that maps a player symbol to a matrix for that player"
    B::NamedTuple{ps, TB}
end
n_states(d::NPlayerLinearDynamics{nx}) where {nx} = nx
n_players(d::NPlayerLinearDynamics{nx, np}) where {nx, np} = np
player_names(n::NPlayerLinearDynamics{nx, np, ps}) where {nx, np, ps} = ps
# TODO there must be a way to get this non-allocating
u_dims(d::NPlayerLinearDynamics) = NamedTuple{player_names(d)}((last(size(Bi)) for Bi in d.B))

"""
$(TYPEDEF)

A struct to represent the quadratic players costs for a singel player in a
game.

# Parameters

- `nx`: the number of states
- `np`: the number of players
- `ps`: a tuple of player symbols
- `T`: the matrix element type (e.g. Flaot64)
- `TR`: the type tuple of R-matrices

# Fields

$(TYPEDFIELDS)
"""
struct QuadraticPlayerCost{nx, np, ps, T, TR<:NTuple{np, SMatrix}}
    "The qudratic state cost matrix"
    Q::SMatrix{nx, nx, T}
    "The linear state cost."
    l::SVector{nx, T}
    "A named tuple that maps a :player to the quadratic control cost matirx,
    that represents the cost that *this* player encounters for :player taking a
    certain control u"
    R::NamedTuple{ps, TR}
end

"""
$(TYPEDEF)

A struct to represent a multi-player player differential game.

# Parameters

- `nx`: the number of states
- `np`: the number of players
- `ps` a tuple of player symbols
- `h`: the horizon of the game (number of steps, Int)
- `TD`: the type of the dynamics representation
- `TC`: the type tuple of cost represntations

# Fields

$(TYPEDFIELDS)
"""
struct NPlayerFiniteHorizonLQGame{nx, np, ps, h,
                                  TD<:NPlayerLinearDynamics{nx, np, ps},
                                  TC<:NTuple{np, QuadraticPlayerCost{nx, np, ps}}}
    "The linear, time varying dynamics of the system"
    dynamics::SVector{h, TD}
    "The quadratic, time varying costs for each player in terms of a vector
    NamedTuples mapping :player to the corresponding cost representation."
    cost::SVector{h, NamedTuple{ps, TC}}
end

n_states(g::NPlayerFiniteHorizonLQGame{nx}) where {nx} = nx
n_players(g::NPlayerFiniteHorizonLQGame{nx, np}) where {nx, np} = np
player_names(g::NPlayerFiniteHorizonLQGame{nx, np, ps}) where {nx, np, ps} = ps
horizon(g::NPlayerFiniteHorizonLQGame{nx, np, ps, h}) where {nx, np, ps, h} = h
# TODO: statically access the control dimensions

"""
$(TYPEDSIGNATURES)

Solve a time-varying, finite horizon LQ-game to find closed-loop NASH feedback
strategies for both players.

Assumes that dynamics are given by `xₖ₊₁ = Aₖ*xₖ + ∑ᵢBₖⁱ uₖⁱ`.

"""

# TODO: NU should be known at compile time from g
function solve_lq_game(g::NPlayerFiniteHorizonLQGame, val_total_u::Val{nu} = Val{2}()) where {nu}
    # the number of controls for every player
    u_idx_cumsum = NamedTuple{player_names(g)}(cumsum(collect(values(u_dims(first(g.dynamics))))))
    # the index range for every player
    u_idx_range = NamedTuple{player_names(g)}([begin
                                                first_idx = (ii == 1 ? 1 : u_idx_cumsum[ii-1] + 1);
                                                last_idx = u_idx_cumsum[ii];
                                                SVector{u_dims(first(g.dynamics))[ii]}(first_idx:last_idx)
                                            end for ii in 1:length(player_names(g))])
    total_u_idx_range = SVector{nu}(1:nu)
    # TODO: maybe also move everything above to game contruction. This only
    # needs to be done once and can even be known at parse-time

    # initializting the optimal cost to go representation for DP
    # quadratic cost to go
    Z = NamedTuple{player_names(g)}(player_cost.Q for player_cost in last(g.cost))
    ζ = NamedTuple{player_names(g)}(player_cost.l for player_cost in last(g.cost))

    # TODO: figure these types out automatically. Also the dimensions might actually not be the same!
    strategies = Vector(undef, horizon(g))
    P_split = Vector(undef, n_players(g))
    α_split = Vector(undef, n_players(g))

    # working backwards in time to solve the dynamic program
    for kk in horizon(g):-1:1
        dyn = g.dynamics[kk]
        cost = g.cost[kk]
        # convenience shorthands for the relevant quantities
        A = dyn.A; B = dyn.B

        # Compute Ps given previously computed Zs.
        # Refer to equation 6.17a in Basar and Olsder.
        # This will involve solving a system of matrix linear equations of the
        # form [S1s; S2s; ...] * [P1; P2; ...] = [Y1; Y2; ...].

        # Setup the S and Y matrix of the S * X = Y matrix equation
        # TODO: optimize! -- this takes too long, nu not known at compile time?
        S = @SMatrix zeros(0, nu)
        Y = @SMatrix zeros(0, n_states(g) + 1)

        # TODO: I think this can be written as an outer product to avoid concatenating
        # So if we had the full vector:
        # S = B' Z B + blkdiag(R)
        # Y = B' Z A
        for ii in 1:n_players(g)
            BᵢZᵢ = B[ii]' * Z[ii]
            udim_ii = last(size(B[ii]))
            # the current set of rows that we construct for player ii
            S_row = @SMatrix zeros(udim_ii, 0)
            # the term for own own control cost
            for jj in 1:n_players(g)
                # append the column for the jth player to the current row
                S_row = hcat(S_row, (ii == jj ? cost[ii].R[ii] + BᵢZᵢ * B[ii] : BᵢZᵢ * B[jj]))
            end
            # append the fully constructed row to the full S-Matrix
            S = vcat(S, S_row)
            Y = vcat(Y, [(BᵢZᵢ*A) (B[ii]'*ζ[ii])])
        end

        # solve for the gains `P` and feed forward terms `α` simulatiously
        P_and_α = S \ Y
        # TODO: optimize! -- this splitting seems to cost a lot of time
        P = P_and_α[:, total_u_idx_range]
        α = P_and_α[:, end]

        for ii in 1:n_players(g)
            P_split[ii] = P[u_idx_range[ii], :]
            α_split[ii] = α[u_idx_range[ii]]
        end

        # compute F and β as intermediate result for estimating the cost to go
        B_row_vec = hcat(B...)
        F = A - B_row_vec * P
        β = -B_row_vec * α

        # update Z and ζ (cost to go representation for the next step backwards
        # in time)
        ζ = NamedTuple{player_names(g)}((F' * (ζ[ii] + Z[ii] * β) + cost[ii].l) + sum(P_split[jj]'*cost[ii].R[jj]*α_split[jj] for jj in 1:n_players(g)) for ii in 1:n_players(g))
        Z = NamedTuple{player_names(g)}(F' * Z[ii] * F + cost[ii].Q + sum(P_split[jj]'*cost[ii].R[jj]*P_split[jj] for jj in 1:n_players(g)) for ii in 1:n_players(g))

        # TODO: optimize! -- there must be something faster than this. Maybe
        # look at how they do it in `DynamicalSystems.jl` and/or `DifferentialEquations.jl`
        strategies[kk] = (P_split, α_split)
    end

    return strategies
end

# TODO: Maybe the input to this should be a named-tuple of structs to properly
# handle the problem of type stability for different input dimensions over
# players
"""
# Fields

- `As`: list over time: of state dynamics matrix Aₖ

- `Bs`: list over time, over players: of control input matrix Bₖⁱ

- `Qs`: list over time, over players: of quadratic state cost Qₖⁱ

- `ls`: list over time, over players: of linear state cost lₖⁱ

- `Rs`: list over time, over `player_a` over `player_b`: of quadratic control cos
        (cost that player a sees if player b takes a certain control action)

"""
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
