function trajectory!(traj::SystemTrajectory{h}, cs::ControlSystem,
                     γ::SizedVector{h, <:AffineStrategy},
                     last_op::SystemTrajectory{h}, x0::SVector,
                     max_elweise_divergence::Float64=Inf) where {h}

    @assert sampling_time(traj) == sampling_time(last_op) == sampling_time(cs)

    xₖ = x0
    # xₖ = first(last_op.x)

    for k in 1:h
        # the quantities on the old operating point
        x̃ₖ = last_op.x[k]
        ũₖ = last_op.u[k]
        # the current strategy
        γₖ = γ[k]
        # the deviation from the last operating point
        Δxₖ = xₖ - x̃ₖ

        if norm(Δxₖ, Inf) > max_elweise_divergence
            return false
        end

        # record the new operating point:
        x_opₖ = traj.x[k] = xₖ
        u_opₖ = traj.u[k] = control_input(γₖ, Δxₖ, ũₖ)

        # integrate x forward in time for the next iteration.
        xₖ = next_x(cs, x_opₖ, u_opₖ, k)
    end
    return true
end

"""
$(FUNCTIONNAME)(g::AbstractGame, traj::SystemTrajectory)

Returns a vector of costs for each player
"""
function cost(g::GeneralGame, traj::SystemTrajectory)
    map(player_index(g)) do i
        sum(player_costs(g)[i](traj.x[k], traj.u[k], sampling_time(dynamics(g))*(k-1)) for k
            in eachindex(traj.x))
    end
end
