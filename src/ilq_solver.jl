@with_kw struct iLQSolver
    "The initial scaling of the feed-forward term."
    α_scale_init::Float64 = 0.1
    "The geometric scaling of the feed-forward term per scaling step in
    backtrack scaling."
    α_scale_step::Float64 = 0.5
    "Iteration is aborted if this number is exceeded."
    max_n_iter::Int = 1000
    "The maximum number of backtrackings per scaling step"
    max_scale_backtrack::Int = 10
    "The maximum elementwise difference bewteen operating points for
    convergence."
    max_elwise_diff_converged::Float64 = 0.1
    "The maximum elementwise difference bewteen operating points for per
    iteration step."
    max_elwise_diff_step::Float64 = 1.
end

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
        scale!(current_strategy, current_op, sf)
        # we compute the new trajectory but abort integration once we have
        # diverged more than solver.max_elwise_diff_step
        if trajectory!(next_op, dynamics(g), current_strategy, current_op,
                       first(current_op.x), solver.max_elwise_diff_step)
            return true, next_op
        end
    end
    # TODO: in this case, the result in next_op is not really meaningful
    return false, next_op
end


"""

    $(FUNCTIONNAME)(p::AbstractGame,
                    initial_op::SystemTrajectory,
                    initial_strategy::StaticVector)

Computes a solution solution to a (potentially non-linear and non-quadratic)
finite horizon game g.
"""
function solve(g::AbstractGame, solver::iLQSolver, x0::SVector,
               initial_op::SystemTrajectory,
               initial_strategy::StaticVector)


    # safe the start time of our computation
    start_time = time()

    i_iter = 0
    # allocate memory for the last and the current operating point
    # TODO: can we allocate this outside this loop?
    last_op = copy(initial_op)
    current_op = initial_op
    current_strategy = initial_strategy
    lqg_approx = LQGame(undef, g, Val(horizon(initial_op)))

    # 0. compute the operating point for the first run.
    trajectory!(current_op, dynamics(g), current_strategy, last_op, x0)

    # ... and upate the current by integrating the non-linear dynamics
    while !has_converged(solver, last_op, current_op, i_iter)
        i_iter += 1
        # sanity chech to make sure that we don't manipulate the wrong
        # object...
        @assert !(last_op === current_op) "current and last operating point
        refer to the *same* object."

        # 1. linearize dynamics and quadratisize costs to obtain an lq game
        lq_approximation!(lqg_approx, g, current_op)

        # 2. solve the current lq version of the game
        current_strategy = solve_lq_game(lqg_approx)

        # 3. do line search to stabilize the strategy selection and extract the
        # next operating point
        last_op = copy(current_op)
        success, current_op = backtrack_scale!(current_strategy, current_op, g, solver)
        if(!success)
            @error "Could not stabilize solution."
        end
    end

    return current_op, current_strategy
end
