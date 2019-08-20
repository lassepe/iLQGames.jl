using Test

include("utils.jl")

using StaticArrays
using iLQGames:
    Lorenz3D,
    dx,
    linearize

l3 = Lorenz3D(10., 28., 8//3)

# test some linearization
@testset "control_system.jl" begin
    @test @inferred_with_info dx(l3, @SVector([2, 1, 7]), @SVector([0, 1]), 0)
    @test @inferred_with_info linearize(l3, @SVector([2, 1, 7]), @SVector([0, 1]), 0)
end
