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

    # initializting the optimal cost to go representation for DP
    # quadratic cost to go
    Z = [pc.Q for pc in last(player_costs(g))]
    ζ = [pc.l for pc in last(player_costs(g))]

    # Setup the S and Y matrix of the S * P = Y matrix equation
    # As `nx` and `nu` are known at compile time and all operations below can be
    # inlined, allocations can be eliminted by the compiler.
    S = @MMatrix zeros(nu, nu)
    YP = @MMatrix zeros(nu, nx)
    Yα = @MVector zeros(nu)

    # working backwards in time to solve the dynamic program
    for kk in horizon(g):-1:1
        dyn = dynamics(g)[kk]
        cost = player_costs(g)[kk]
        # convenience shorthands for the relevant quantities
        A = dyn.A
        B = dyn.B

        # Compute Ps given previously computed Zs.
        # Refer to equation 6.17a in Basar and Olsder.
        # This will involve solving a system of linear matrix equations of the
        # form S * P = Y.
        for (ii, udxᵢ) in enumerate(uindex(g))
            BᵢZᵢ = B[:, udxᵢ]' * Z[ii]
            # the current set of rows that we construct for player ii
            S[udxᵢ, :] = cost[ii].R[udxᵢ, :] + BᵢZᵢ*B
            # append the fully constructed row to the full S-Matrix
            YP[udxᵢ, :] = BᵢZᵢ*A
            Yα[udxᵢ] = B[:, udxᵢ]'*ζ[ii] + cost[ii].r[udxᵢ]
        end

        Sinv = inv(SMatrix(S))
        P = Sinv * SMatrix(YP)
        α = Sinv * SVector(Yα)

        # compute F and β as intermediate result for estimating the cost to go
        F = A - B * P
        β = -B * α

        # update Z and ζ (cost to go representation for the next step backwards
        # in time)
        for ii in 1:n_players(g)
            cᵢ= cost[ii]
            PRᵢ = P' * cᵢ.R
            ζ[ii] = F' * (ζ[ii] + Z[ii] * β) + cᵢ.l + PRᵢ * α - P' * cᵢ.r
            Z[ii] = F' * Z[ii] * F + cᵢ.Q + PRᵢ * P
        end

        strategies[kk] = AffineStrategy(P, α)
    end
end
