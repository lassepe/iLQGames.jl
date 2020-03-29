function trajectory!(traj::SystemTrajectory{h}, cs::ControlSystem,
                     γ::AbstractVector{<:AffineStrategy},
                     last_op::SystemTrajectory{h}, x0::AbstractVector,
                     max_elwise_divergence::Float64=Inf) where {h}

    @assert samplingtime(traj) == samplingtime(last_op) == samplingtime(cs)

    xₖ = x0

    for k in 1:h
        tₖ = time_disc2cont(traj, k)
        # the quantities on the old operating point
        x̃ₖ = last_op.x[k]
        ũₖ = last_op.u[k]
        # the current strategy
        γₖ = γ[k]
        # the deviation from the last operating point
        Δxₖ = xₖ - x̃ₖ

        # return early if divergence from operating point exceeded
        infnorm(Δxₖ) <= max_elwise_divergence || return false

        # record the new operating point:
        traj.x[k] = xₖ
        traj.u[k] = uₖ = control_input(γₖ, Δxₖ, ũₖ)

        # integrate x forward in time for the next iteration.
        xₖ = next_x(cs, xₖ, uₖ, tₖ)
    end
    return true
end

"""
$(FUNCTIONNAME)(g::AbstractGame, traj::SystemTrajectory)

Returns a vector of costs for each player
"""
function cost(g::GeneralGame, traj::SystemTrajectory)
    map(player_costs(g)) do ci
        sum(ci(g, traj.x[k], traj.u[k], time_disc2cont(traj, k)) for k in
            eachindex(traj.x))
    end
end
