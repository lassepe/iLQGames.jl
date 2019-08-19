using Revise
using DocStringExtensions

using LinearAlgebra
using StaticArrays

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
function solve_lq_game(As::AbstractVector,
                       Bs::AbstractVector,
                       Qs::AbstractVector,
                       ls::AbstractVector,
                       Rs::AbstractVector)
    horizon = length(As)
    num_players = length(first(Bs))
    total_xdim = first(size(first(As)))
    # the number of controls for every player
    u_dims = [last(size(Bi)) for Bi in first(Bs)]
    u_idx_cumsum = cumsum(u_dims)
    # the index range for every player
    # TODO: maybe use this in more places
    u_idx_range = map(1:num_players) do ii
        first_idx = ii == 1 ? 1 : u_idx_cumsum[ii-1] + 1
        last_idx = u_idx_cumsum[ii]
        return first_idx:last_idx
    end
    total_udim = sum(u_dims)

    # initializting the optimal ocst to go representation for dynamic
    # programming
    # quadratic cost to go
    Z = last(Qs)
    # linear cost to go
    ζ = last(ls)

    strategies = []

    # working backwards in time to solve the dynamic program
    for kk in horizon:-1:1
        # convenience shorthands for the relevant quantities
        A = As[kk]; B = Bs[kk]; Q = Qs[kk]; l = ls[kk]; R = Rs[kk];

        # Compute Ps given previously computed Zs.
        # Refer to equation 6.17a in Basar and Olsder.
        # This will involve solving a system of matrix linear equations of the
        # form [S1s; S2s; ...] * [P1; P2; ...] = [Y1; Y2; ...].

        # Setup the S and Y matrix of the S * X = Y matrix equation
        S = zeros(0, total_udim)
        Y = zeros(0, total_xdim + 1)

        # TODO maybe optimize this to allow for SMatrix or at least MMatrix.
        # Maybe concatenating is the better thing to do here if things are
        # static?
        for ii in 1:num_players
            BᵢZᵢ = B[ii]' * Z[ii]
            udim_ii = last(size(B[ii]))
            # the current set of rows that we construct for player ii
            S_row = zeros(udim_ii, 0)
            # the term for own own control cost
            for jj in 1:num_players
                # TODO: maybe think about col-major optimization here for
                # caching or windowing to avoid concatenating
                # append the column for the jth player to the current row
                S_row = hcat(S_row, (ii == jj ? R[ii][ii] + BᵢZᵢ * B[ii] : BᵢZᵢ * B[jj]))
            end
            # append the fully constructed row to the full S-Matrix
            S = vcat(S, S_row)
            Y = vcat(Y, [(BᵢZᵢ*A) (B[ii]'*ζ[ii])])
        end

        # solve for the gains `P` and feed forward terms `α` simulatiously
        P_and_α = S \ Y
        P = P_and_α[:, 1:total_udim]
        P_split = [P[u_idx_range[ii], :] for ii in 1:num_players]
        α = P_and_α[:, end]
        α_split = [α[u_idx_range[ii]] for ii in 1:num_players]

        # compute F and β as intermediate result for estimating the cost to go
        # for the next step backwards in time
        # TODO: the splat operator here might be really slow
        B_row_vec = hcat(Bs...)
        F = A - B_row_vec * P
        β = -B_row_vec * α

        # update Z and ζ (cost to go representation for the next step backwards
        # in time)
        for ii in 1:num_players
            ζ[ii] = F' * (ζ[ii] + Z[ii] * β) + l[ii] + sum(P_split[jj]' * R[ii][jj] * α_split[jj] for jj in 1:num_players)
            # TODO (maybe use smart indexing, offset arrays or subarrays for this)
            # Also this should be expressablea s one large matrix equation without any explicit sum
            Z[ii] = F' * Z[ii] * F + Q[ii] + sum(P_split[jj]' * R[ii][jj] * P_split[jj] for jj in 1:num_players)
        end

        pushfirst!(strategies, (P_split, α_split))
    end

    return strategies
end


# solve two-player inifinite horizion (time-invariant) LQ game by Lyapunov
# iterations
function solve_lyapunov_iterations(A::AbstractArray, B1::AbstractArray, B2::AbstractArray,
                                   Q1::AbstractArray, Q2::AbstractArray,
                                   R11::AbstractArray, R12::AbstractArray,
                                   R21::AbstractArray, R22::AbstractArray, n_iter::Int=100)

    # Initialize cost-to-go with terminal state cost
    Z1 = Q1
    Z2 = Q2

    # Initialize controls
    P1 = (R11 + B1' * Z1 * B1) \ (B1' * Z1 * A)
    P2 = (R22 + B2' * Z2 * B2) \ (B2' * Z2 * A)

    for ii in 1:n_iter
        P1_old = P1
        P2_old = P2

        P1 = (R11 + B1' * Z1 * B1) \ (B1' * Z1 * (A - B2 * P2_old))
        P2 = (R22 + B2' * Z2 * B2) \ (B2' * Z2 * (A - B1 * P1_old))

        Z1 = (A - B1 * P1 - B2 * P2)' * Z1 * (A - B1 * P1 - B2 * P2) + P1' * R11 * P1 + P2' * R12 * P2 + Q1
        Z2 = (A - B1 * P1 - B2 * P2)' * Z2 * (A - B1 * P1 - B2 * P2) + P1' * R21 * P1 + P2' * R22 * P2 + Q2
    end

    return (P1, P2)
end

# Testing the solver at a simple example: A two-player point mass 1D system.

# The state composes of position and oritation. Therefore, the system dynamics
# are a pure integrator.
ΔT = 0.1
H = 10.0
N_STEPS = Int(H / ΔT)

# contiuous time system
#      A = [0 1; 0 0]
#      B1 = [0.05, 1.0]
#      B2 = [0.032, 0.11]
#
#      # discrete version (See trick
#      # https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models)
#
#      # TODO: there still seems to be some numerical issue with the linearization
#      eABT = exp([A B1 B2; zeros(2, 4)] * ΔT)
#      A_disc = SMatrix{2,2}(eABT[1:2, 1:2])
#      B1_disc = SMatrix{2,1}(eABT[1:2, 3])
#      B2_disc = SMatrix{2,1}(eABT[1:2, 4])
A_disc = SMatrix{2, 2}([1.0 ΔT; 0.0 1.0])
B1_disc = SMatrix{2, 1}([0.5 * ΔT^2, ΔT])
B2_disc = SMatrix{2, 1}([0.32 * ΔT^2, 0.11 * ΔT])

# state cost
Q1 = @SMatrix [1. 0.; 0. 1.]
Q2 = -Q1
l1 = @SVector zeros(2)
l2 = -l1

# control cost
R11 = @SMatrix [1.]
R12 = @SMatrix [0.]
R21 = @SMatrix [0.]
R22 = @SMatrix [1.]

# sequence for the finite horizon
As = repeat([A_disc], N_STEPS)
Bs = repeat([[B1_disc, B2_disc]], N_STEPS)
Qs = repeat([[Q1, Q2]], N_STEPS)
ls = repeat([[l1, l2]], N_STEPS)
Rs = repeat([[[R11, R12], [R21, R22]]], N_STEPS)

# Lyapunov test:


# 1.
# First let's solve for the strategies when using Lyapunov
# Benchmark to see whethe we are doing useless memory allocation. Looks good!
# @benchmark solve_lyapunov_iterations($A_disc, $B1_disc, $B2_disc, $Q1, $Q2,
#                                      $R11, $R12, $R21, $R22)
# Actual Run
P1, P2 = solve_lyapunov_iterations(A_disc, B1_disc, B2_disc, Q1, Q2, R11, R12,
                                   R21, R22)

@info "Lyapunov solution P1"
show(P1)
# NOTE: comapring this to the cpp implementation we get approximtaely the same
# solution as the cpp lyampunov

# 2.
# Now let's to the same thing with the finite horizoin solver. We would expect
# the gains of the strategies at the beginning of the game to be similar to the
# Lyapunov inifinite horizoin solution.
strategies = solve_lq_game(As, Bs, Qs, ls, Rs)
println("")
@info "LQGame solution P1"
show(strategies |> first |> first |> first)

# TODO: it is correct that the alphas are zero but the Ps converge to the wrong thing.
