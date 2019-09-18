# Some Simple Tests
using Test

using Pkg; Pkg.add(PackageSpec(url="https://github.com/lassepe/ProfileTools.jl"))

macro testset_include(filename)
    @assert filename isa AbstractString
    quote
        @testset $filename begin
            include($filename)
        end;
    end
end

# TODO this could pontially do a glob on all `test_` files
@testset "all" begin
    @testset_include "test_control_system.jl"
    @testset_include "test_linear_system.jl"
    @testset_include "test_solve_lq_game.jl"
    @testset_include "test_ilq_solver.jl"
end;
