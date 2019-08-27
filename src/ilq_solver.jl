using DocStringExtensions

include("finite_horizon_lq_game.jl")



# TODO tidy up and maybe extracts constants into solver
function has_converged(last_op::StaticVector{h}, current_op::StaticVector{h},
                       n_iter::Int) where {h}
    max_n_iter = 100
    max_elwise_diff = 0.01

    if n_iter == 0
        return false
    elseif n_iter >= max_n_iter
        return true
    end

    # TODO: this might be very slow, depending on what this is lowered to
    return any(norm(p1.x - p2.x, Inf) > max_elwise_diff
               for (p1, p2) in zip(current_op, last_op))
end

# TODO implement
# # TODO maybe the game g shoudl hold the operating point (to manage memory
# allocation)
# updates the operating point
function update_op!(current_op, g::FiniteHorizonGame, last_op, current_strategy)
    # integrate through full dynamics of the game by applying the current
    # strategy at the last_op. New state trajectory compreses the new operating
    # point
    @error "not implemented"
    return current_op
end

# TODO: implement
# TODO: there must be a better name for this
# modifies the current strategy to stabilize the update
function modify_strategy!(current_strategy, current_op)
    @error "not implemented"
    return current_strategy
end

"""

    $(FUNCTIONNAME)(p::FiniteHorizonGame, initial_operating_point::StaticVector,
                   initial_strategy::StaticVector, max_runtime_seconds::Real)

Computes a solution solution to a (potentially non-linear and non-quadratic)
finite horizon game g.

TODO: refine once implemented
"""
# TODO: maybe x0 should be part of the problem (of a nonlinear problem struct)
function solve(g::FiniteHorizonGame, x0:: initial_operating_point::StaticVector,
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
        update_op!(current_op, g, last_op, current_strategy)

        # 2. linearize dynamics and quadratisize costs to obtain an lq game
        # TODO: implement
        # TODO: maybe do this in-place
        lqg_approx = lq_approximation(g, current_op)

        # 3. solve the current lq version of the game
        current_strategy = solve_lq_game(lqg_approx)

        # 4. do line search to stabilize the strategy selection
        modify_strategy!(current_strategy, current_op)
    end

    return current_op, current_strategy
end
