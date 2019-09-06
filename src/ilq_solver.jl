@with_kw struct iLQSolver
    "The scaling of the feed-forward term."
    α_scaling::Float64 = 0.001
    "Iteration is aborted if this number is exceeded."
    max_n_iter::Int = 100
    "The maximum elementwise difference bewteen the current and the last
    operating state trajectory to consider the probem converged."
    max_elwise_diff::Float64 = 0.1
    "The maximum runtime after which iteration is aborted."
    max_runtime_seconds::Float64 = 1.
end

# TODO tidy up and maybe extracts constants into solver
function has_converged(solver::iLQSolver,
                       last_op::SystemTrajectory{h},
                       current_op::SystemTrajectory{h},
                       i_iter::Int) where {h}
    if i_iter == 0
        return false
    elseif i_iter >= solver.max_n_iter
        @warn "Iteration aborted because max interations exceeded."
        return true
    end

    # TODO: this might be very slow, depending on what this is lowered to
    println(length(last_op.x))
    @show maximum(norm(current_op.x[k] - last_op.x[k], Inf) for k in eachindex(last_op.x))
    return all(norm(current_op.x[k] - last_op.x[k], Inf) < solver.max_elwise_diff for k in eachindex(last_op.x))
end

# TODO: there must be a better name for this
# modifies the current strategy to stabilize the update
function stabilize!(current_strategy::SizedVector, solver::iLQSolver, current_op::SystemTrajectory)
    # TODO: implement this backtracking search
    map!(current_strategy, current_strategy) do el
        return AffineStrategy(el.P, el.α * solver.α_scaling)
    end
    return true
end

"""

    $(FUNCTIONNAME)(p::AbstractGame,
                    initial_op::SystemTrajectory,
                    initial_strategy::StaticVector)

Computes a solution solution to a (potentially non-linear and non-quadratic)
finite horizon game g.

TODO: refine once implemented
"""
# TODO: maybe x0 should be part of the problem (of a nonlinear problem struct)
function solve(g::AbstractGame, solver::iLQSolver, x0::SVector,
               initial_op::SystemTrajectory,
               initial_strategy::StaticVector)


    # safe the start time of our computation
    start_time = time()
    # TODO: magic number (the guessed duration of one iteration)
    has_time_remaining() = time() - start_time + 0.02 < solver.max_runtime_seconds

    # TODO: depending on what will happen, we need to explicitly copy or use
    # `similar` here
    i_iter = 0
    # allocate memory for the last and the current operating point
    # TODO: try not to copy!
    last_op = initial_op
    current_op = initial_op
    current_strategy = initial_strategy

    while !has_converged(solver, last_op, current_op, i_iter)
        i_iter += 1
        # 1. cache the current operating point ...
        # # TODO: this is really annoying -- avoid deepcopy
        # at least overload the copy! interface
        last_op = deepcopy(current_op)
        # ... and upate the current by integrating the non-linear dynamics
        trajectory!(current_op, dynamics(g), current_strategy, last_op, x0)

        # 2. linearize dynamics and quadratisize costs to obtain an lq game
        # TODO: implement
        # TODO: maybe do this in-place
        lqg_approx = lq_approximation(g, current_op)

        # 3. solve the current lq version of the game
        current_strategy = solve_lq_game(lqg_approx)

        # 4. do line search to stabilize the strategy selection
        if(!stabilize!(current_strategy, solver, current_op))
            @error "Could not stabilize solution."
        end
    end

    @info "Finished after $i_iter interations."

    return current_op, current_strategy
end
