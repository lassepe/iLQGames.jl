struct iLQSolver
    "The scaling of the feed-forward term."
    α_scaling::Float64
    "The maximum number of iterations. Iteration is aborted if this number is
    exceeded."
    max_n_iter::Int
    "The maximum elementwise difference bewteen the current and the last
    operating state trajectory to consider the probem converged."
    max_elwise_diff::Float64
    "The time step for conversion of the problem from contiuous to discrete
    time."
    ΔT::Float64
end

# TODO tidy up and maybe extracts constants into solver
function has_converged(solver::iLQSolver, last_op::StaticVector{h}, current_op::StaticVector{h},
                       n_iter::Int) where {h}
    if n_iter == 0
        return false
    elseif n_iter >= solver.max_n_iter
        return true
    end

    # TODO: this might be very slow, depending on what this is lowered to
    return any(norm(p1.x - p2.x, Inf) > solver.max_elwise_diff
               for (p1, p2) in zip(current_op, last_op))
end

# # TODO maybe the game g or a solver should hold the operating point (to
# manage memory allocation) updates the operating point

# Integrate through full dynamics of the game by applying the current strategy
# at the last operating point. From this, we obtain a new state and control
# trajectory (vector (over time) of tuples (x, u))
function update_op!(current_op::SizedVector{h,<:Tuple}, g::FiniteHorizonGame,
                    solver::iLQSolver, last_op::SizedVector{h,<:Tuple},
                    current_strategy::SizedVector{h,<:AffineStrategy}) where {h}
    xₖ,_ = first(last_op)
    for k in 1:h
        # the quantities on the old operating point
        x̃ₖ, ũₖ = last_op[k]
        # the current strategy
        γₖ = current_strategy[k]
        # the deviation from the last operating point
        Δxₖ = x - x̃ₖ

        # record the new operating point:
        x_opₖ = x
        u_opₖ = control_input(γₖ, Δxₖ, ũₖ)
        current_op = (x_opₖ, u_opₖ)

        # integrate x forward in time for the next iteration.
        x = integrate(dynamics(g), x_opₖ, u_opₖ, 0, solver.ΔT)
    end
    return current_op
end

# TODO: there must be a better name for this
# modifies the current strategy to stabilize the update
function stabilize!(current_strategy, solver::iLQSolver, current_op::SizedVector{<:Tuple})
    # TODO: implement this backtracking search
    map!(current_strategy) do elₖ
        Pₖ, αₖ = elₖ
        return (Pₖ, αₖ * solver.α_scaling)
    end
    return true
end

"""

    $(FUNCTIONNAME)(p::FiniteHorizonGame,
                    initial_operating_point::StaticVector,
                    initial_strategy::StaticVector, max_runtime_seconds::Real)

Computes a solution solution to a (potentially non-linear and non-quadratic)
finite horizon game g.

TODO: refine once implemented
"""
# TODO: maybe x0 should be part of the problem (of a nonlinear problem struct)
function solve(g::FiniteHorizonGame, solver::iLQSolver, x0::SVector
               initial_operating_point::StaticVector,
               initial_strategy::StaticVector, max_runtime_seconds::Real)


    # safe the start time of our computation
    start_time = time()
    has_time_remaining() = time() - start_time + 0.02 < max_runtime_seconds

    # TODO: depending on what will happen, we need to explicitly copy or use
    # `similar` here
    num_iterations = 0
    # allocate memory for the last and the current operating point
    last_op = similar(initial_operating_point)
    current_op = initial_operating_point
    current_strategy = initial_strategy

    while (has_time_remaining() && !has_converged())
        num_iterations += 1
        # 1. cache the current operating point ...
        last_op = current_op
        # ... and upate the current by integrating the non-linear dynamics
        update_op!(current_op, g, solver, last_op, current_strategy)

        # 2. linearize dynamics and quadratisize costs to obtain an lq game
        # TODO: implement
        # TODO: maybe do this in-place
        lqg_approx = lq_approximation(g, current_op)

        # 3. solve the current lq version of the game
        current_strategy = solve_lq_game(lqg_approx)

        # 4. do line search to stabilize the strategy selection
        if(!stabilize!(current_strategy, solver, current_op))
            @error "Did not find a solution!"
        end
    end

    return current_op, current_strategy
end
