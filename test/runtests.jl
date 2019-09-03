# Some Simple Tests
using Test

include("utils.jl")

# TODO this could pontially do a glob on all `test_` files

@testset "all" begin
    @testset_include "test_control_system.jl"
    @testset_include "test_linear_system.jl"
    @testset_include "test_solve_lq_game.jl"
#    @testset_include "test_ilq_solver.jl"
end;
