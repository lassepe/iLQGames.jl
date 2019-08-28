# Some Simple Tests
using Test

include("utils.jl")

@testset "all" begin
    @testset_include "test_control_system.jl"
    @testset_include "test_linear_system.jl"
    @testset_include "test_solve_lq_game.jl"
end;
