using Test

using iLQGames:
    LinearSystem,
    discretize,
    discretize_exp,
    linearize,
    linearize_discrete

using StaticArrays

A = @SMatrix [1 2; 3 4]
B = @SMatrix [1; 3]
ls = LinearSystem(A, B)
ΔT = 0.1241241

# some test point
x0 = @SVector [1., 2.]
u0 = @SVector [2.]
t0 = .11

d1 = discretize(ls, ΔT)
d2 = discretize_exp(ls, ΔT)

@testset "discretize vs. discretize_exp" begin
    @test isapprox(d1.A, d2.A)
    @test isapprox(d1.B, d2.B)
end;

@testset "linearize" begin
    # linearizing a linear system should not make a difference
    ls_lin = linearize(ls, x0, u0, t0)
    @test ls.A == ls_lin.A
    @test ls.B == ls_lin.B
end;

@testset "linearize_discrete" begin
    # calling linearize_discrete on a linear system should give the same
    # as discretize alone
    ls_lind = linearize_discrete(ls, x0, u0, t0, ΔT)
    @test isapprox(d1.A, ls_lind.A)
    @test isapprox(d1.B, ls_lind.B)
end
