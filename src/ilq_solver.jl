@with_kw struct iLQSolver
    "The initial scaling of the feed-forward term."
    α_scale_init::Float64 = 0.015
    "The geometric scaling of the feed-forward term per scaling step in
    backtrack scaling."
    α_scale_step::Float64 = 0.5
    "Iteration is aborted if this number is exceeded."
    max_n_iter::Int = 1000
    "The maximum number of backtrackings per scaling step"
    max_scale_backtrack::Int = 10
    "The maximum elementwise difference bewteen operating points for
    convergence."
    max_elwise_diff_converged::Float64 = 0.01
    "The maximum elementwise difference bewteen operating points for per
    iteration step."
    max_elwise_diff_step::Float64 = 1.
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

    return are_close(current_op, last_op, solver.max_elwise_diff_converged)
end

function are_close(op1::SystemTrajectory, op2::SystemTrajectory,
                   max_elwise_diff::Float64)
    @assert horizon(op1) == horizon(op2)
    return all(norm(op1.x[k] - op2.x[k], Inf) < max_elwise_diff for k in
               eachindex(op2.x))
end

# modifies the current strategy to stabilize the update
function scale!(current_strategy::SizedVector, current_op::SystemTrajectory,
                α_scale::Float64)
    map!(current_strategy, current_strategy) do el
        return AffineStrategy(el.P, el.α * α_scale)
    end
end

function backtrack_scale!(current_strategy::SizedVector,
                          current_op::SystemTrajectory, g::AbstractGame,
                          solver::iLQSolver)
    next_op = zero(current_op)
    for i in 1:solver.max_scale_backtrack
        # initially we do a large scaling. Afterwards, always half feed forward
        # term.
		sf = i == 1 ? solver.α_scale_init : solver.α_scale_step
        # TODO: potentially we could use the operating point for next round
        scale!(current_strategy, current_op, sf)
        trajectory!(next_op, dynamics(g), current_strategy, current_op, first(current_op.x))
        if are_close(next_op, current_op, solver.max_elwise_diff_step)
            return true
        end
    end
    return false
end


"""

    $(FUNCTIONNAME)(p::AbstractGame,
                    initial_op::SystemTrajectory,
                    initial_strategy::StaticVector)

Computes a solution solution to a (potentially non-linear and non-quadratic)
finite horizon game g.

TODO: refine once implemented
"""
function solve(g::AbstractGame, solver::iLQSolver, x0::SVector,
               initial_op::SystemTrajectory,
               initial_strategy::StaticVector)


    # safe the start time of our computation
    start_time = time()

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

        @assert !(last_op === current_op) "operating point never even changed"

        # 2. linearize dynamics and quadratisize costs to obtain an lq game
        # TODO: maybe do this in-place
        lqg_approx = lq_approximation(g, current_op)

        # 3. solve the current lq version of the game
        current_strategy = solve_lq_game(lqg_approx)

        # 4. do line search to stabilize the strategy selection
        if(!backtrack_scale!(current_strategy, current_op, g, solver))
            @error "Could not stabilize solution."
        end
    end

    @info "Finished after $i_iter interations."

    return current_op, current_strategy
end
