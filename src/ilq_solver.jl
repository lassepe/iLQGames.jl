@with_kw struct iLQSolver{TLM}
    "The initial scaling of the feed-forward term."
    α_scale_init::Float64 = 0.1
    "The geometric scaling of the feed-forward term per scaling step in
    backtrack scaling."
    α_scale_step::Float64 = 0.5
    "Iteration is aborted if this number is exceeded."
    max_n_iter::Int = 200
    "The maximum number of backtrackings per scaling step"
    max_scale_backtrack::Int = 20
    "The maximum elementwise difference bewteen operating points for
    convergence."
    max_elwise_diff_converged::Float64 = α_scale_init/2
    "The maximum elementwise difference bewteen operating points for per
    iteration step."
    max_elwise_diff_step::Float64 = 20 * max_elwise_diff_converged
    "Preallocated memory for lq approximations."
    _lq_mem::TLM
end

function iLQSolver(g, args...; kwargs...)
    return iLQSolver(args...; kwargs..., _lq_mem=LQGame(undef, g))
end

function has_converged(solver::iLQSolver, last_op::SystemTrajectory{h},
                       current_op::SystemTrajectory{h}) where {h}
    return are_close(current_op, last_op, solver.max_elwise_diff_converged)
end

function are_close(op1::SystemTrajectory, op2::SystemTrajectory,
                   max_elwise_diff::Float64)
    @assert horizon(op1) == horizon(op2)
    return all(infnorm(op1.x[k] - op2.x[k]) < max_elwise_diff for k in
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
    # in this case, the result in next_op is not really meaningful because the
    # integration has not been finished, thus the `success` state needs to be
    # evaluated and handled by the caller.
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
               initial_strategy::StaticVector, verbose::Bool=false)

    converged = false
    i_iter = 0
    # allocate memory for the last and the current operating point
    # TODO: can we allocate this outside this loop? Maybe the solver can manage this
    # memory.
    last_op = copy(initial_op)
    current_op = initial_op
    current_strategy = initial_strategy
    lqg_approx = solver._lq_mem

    # 0. compute the operating point for the first run.
    # TODO -- we could probably allow to skip this in some warm-starting scenarios
    trajectory!(current_op, dynamics(g), current_strategy, last_op, x0)

    # ... and upate the current by integrating the non-linear dynamics
    while !(converged || i_iter >= solver.max_n_iter)
        # sanity chech to make sure that we don't manipulate the wrong
        # object...
        @assert !(last_op === current_op) "current and last operating point
        refer to the *same* object."

        # 1. linearize dynamics and quadratisize costs to obtain an lq game
        lq_approximation!(lqg_approx, g, current_op)

        # 2. solve the current lq version of the game
        solve_lq_game!(current_strategy, lqg_approx)

        # 3. do line search to stabilize the strategy selection and extract the
        # next operating point
        last_op = current_op
        success, current_op = backtrack_scale!(current_strategy, current_op, g, solver)
        if(!success)
            verbose && @warn "Could not stabilize solution."
            # we immetiately return and state that the solution has not been
            # stabilized
            return false, current_op, current_strategy
        end

        i_iter += 1
        converged = has_converged(solver, last_op, current_op)
    end

    # NOTE: for `converged == false` the result may not be meaningful. `converged`
    # has to be handled outside this function
    return converged, current_op, current_strategy
end
