@with_kw struct iLQSolver{TLM, TOM, TQM}
    "The regularization term for the state cost quadraticization."
    state_regularization::Float64 = 0.0
    "The regularization term for the control cost quadraticization."
    control_regularization::Float64 = 0.0
    "The initial scaling of the feed-forward term."
    α_scale_init::Float64 = 0.5
    "The geometric scaling of the feed-forward term per scaling step in
    backtrack scaling."
    α_scale_step::Float64 = 0.5
    "Iteration is aborted if this number is exceeded."
    max_n_iter::Int = 200
    "The maximum number of backtrackings per scaling step"
    max_scale_backtrack::Int = 20
    "The maximum elementwise difference bewteen operating points for
    convergence."
    max_elwise_diff_converged::Float64 = α_scale_init/10
    "The maximum elementwise difference bewteen operating points for per
    iteration step."
    max_elwise_diff_step::Float64 = 30 * max_elwise_diff_converged
    "Preallocated memory for lq approximations."
    _lq_mem::TLM
    "Preallocated memory for quadraticization results."
    _qcache_mem::TQM
    "Preallocated memory for operting points."
    _op_mem::TOM
end

qcache(solver::iLQSolver) = solver._qcache_mem

function regularize(solver::iLQSolver, c::QuadraticPlayerCost)
    return QuadraticPlayerCost(c.l, c.Q + I * solver.state_regularization,
                               c.r, c.R + I * solver.control_regularization)
end

function iLQSolver(g, args...; kwargs...)
    return iLQSolver(args...; kwargs...,
                     _lq_mem=lqgame_preprocess_alloc(g),
                     _qcache_mem=zero(QuadCache{n_states(g),n_controls(g)}),
                     _op_mem=zero(SystemTrajectory, g))
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
                          current_op::SystemTrajectory, last_op::SystemTrajectory,
                          cs::ControlSystem, solver::iLQSolver)
    for i in 1:solver.max_scale_backtrack
        # initially we do a large scaling. Afterwards, always half feed forward
        # term.
		sf = i == 1 ? solver.α_scale_init : solver.α_scale_step
        scale!(current_strategy, current_op, sf)
        # we compute the new trajectory but abort integration once we have
        # diverged more than solver.max_elwise_diff_step
        if trajectory!(current_op, cs, current_strategy, last_op,
                       first(last_op.x), solver.max_elwise_diff_step)
            return true
        end
    end
    # in this case, the result in next_op is not really meaningful because the
    # integration has not been finished, thus the `success` state needs to be
    # evaluated and handled by the caller.
    return false
end

function solve(g::AbstractGame, solver::iLQSolver, args...)
    op0 = zero(SystemTrajectory, g)
    γ0 = zero(strategytype(g))
    return solve!(op0, γ0, g, solver, args...)
end

function solve(initial_strategy::StaticVector, g, args...)
    return solve!(zero(SystemTrajectory, g), copy(initial_strategy), g, args...)
end

function solve(initial_op::SystemTrajectory, initial_strategy::StaticVector, g,
               args...)
    return solve!(copy(initial_op), copy(initial_strategy), g, args...)
end

"Copy the `initial_op` to a new operting point instance using preallocated memory."
function copyop_prealloc!(solver::iLQSolver, initial_op::SystemTrajectory)
    om = solver._op_mem
    new_op = SystemTrajectory{samplingtime(om)}(om.x, om.u, initialtime(initial_op))
    return copyto!(new_op, initial_op)
end

"""

    $(TYPEDSIGNATURES)

Computes a solution solution to a (potentially non-linear and non-quadratic)
finite horizon game g.
"""
function solve!(initial_op::SystemTrajectory, initial_strategy::StaticVector,
                g::GeneralGame, solver::iLQSolver, x0::SVector,
                verbose::Bool=false)

    converged = false
    i_iter = 0
    last_op = copyop_prealloc!(solver, initial_op)

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
        lq_approximation!(lqg_approx, solver, g, current_op)

        # 2. solve the current lq version of the game
        solve_lq_game!(current_strategy, lqg_approx)

        # 3. do line search to stabilize the strategy selection and extract the
        # next operating point
        copyto!(last_op, current_op)
        success = backtrack_scale!(current_strategy, current_op, last_op,
                                   dynamics(g), solver)
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
