"""
$(TYPEDSIGNATURES)

Solve a time-varying, finite horizon LQ-game to find closed-loop NASH feedback
strategies for both players.

Assumes that dynamics are given by `xₖ₊₁ = Aₖ*xₖ + ∑ᵢBₖⁱ uₖⁱ`.

"""
function solve_lq_game!(strategies, g::LQGame)
    # extract control and input dimensions
    nx = n_states(g)
    nu = n_controls(g)

    full_urange = SVector{nu}(1:nu)
    full_xrange = SVector{nx}(1:nx)

    # initializting the optimal cost to go representation for DP
    # quadratic cost to go
    Z = [pc.Q for pc in last(player_costs(g))]
    ζ = [pc.l for pc in last(player_costs(g))]

    # Setup the S and Y matrix of the S * X = Y matrix equation

    S = @MMatrix zeros(nu, nu)
    Y = @MMatrix zeros(nu, nx + 1)

    # working backwards in time to solve the dynamic program
    for kk in horizon(g):-1:1
        dyn = dynamics(g)[kk]
        cost = player_costs(g)[kk]
        # convenience shorthands for the relevant quantities
        A = dyn.A
        B = dyn.B

        # Compute Ps given previously computed Zs.
        # Refer to equation 6.17a in Basar and Olsder.
        # This will involve solving a system of matrix linear equations of the
        # form [S1s; S2s; ...] * [P1; P2; ...] = [Y1; Y2; ...].
        for (ii, udxᵢ) in enumerate(uindex(g))
            BᵢZᵢ = B[:, udxᵢ]' * Z[ii]
            # the current set of rows that we construct for player ii
            S[udxᵢ, :] = cost[ii].R[udxᵢ, full_urange] + BᵢZᵢ*B
            # append the fully constructed row to the full S-Matrix
            Y[udxᵢ, :] = [(BᵢZᵢ*A) (B[:, udxᵢ]'*ζ[ii])]
        end

        # solve for the gains `P` and feed forward terms `α` simulatiously
        P_and_α = SMatrix(S)\SMatrix(Y)
        P = P_and_α[:, full_xrange]
        α = P_and_α[:, end]

        # compute F and β as intermediate result for estimating the cost to go
        F = A - B * P
        β = -B * α

        # update Z and ζ (cost to go representation for the next step backwards
        # in time)
        for ii in 1:n_players(g)
            cᵢ= cost[ii]
            PRᵢ = P' * cᵢ.R
            ζ[ii] = (F' * (ζ[ii] + Z[ii] * β) + cᵢ.l) + PRᵢ * α
            Z[ii] = (F' * Z[ii] * F + cost[ii].Q) + PRᵢ * P
        end

        strategies[kk] = AffineStrategy(P, α)
    end
end
