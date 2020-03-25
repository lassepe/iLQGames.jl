using Test

using iLQGames:
    ProductSystem,
    Unicycle4D,
    n_states,
    n_controls,
    dx,
    linearize

using StaticArrays

# create two separate dynamics systems
unicycle1 = Unicycle4D{0.1}()
unicycle2 = Unicycle4D{0.1}()
xid1 = SVector{4}(1:4)
xid2 = SVector{4}(5:8)
uid1 = SVector{2}(1:2)
uid2 = SVector{2}(3:4)

# and combine them in the product system
dyn = ProductSystem((unicycle1, unicycle2))

# test some operations on the joint system
x = @SVector rand(n_states(dyn))
u = @SVector rand(n_controls(dyn))
t = 0.0;

@testset "dx" begin
    dx_dyn = dx(dyn, x, u, t)
    dx_unicycle1 = dx(unicycle1, x[xid1], u[uid1], t)
    dx_unicycle2 = dx(unicycle2, x[xid2], u[uid2], t)

    @test isapprox(dx_dyn[xid1], dx_unicycle1)
    @test isapprox(dx_dyn[xid2], dx_unicycle2)
end

@testset "linearize" begin
    lin_dyn = linearize(dyn, x, u, t)
    lin_unicycle1 = linearize(unicycle1, x[xid1], u[uid1], t)
    lin_unicycle2 = linearize(unicycle2, x[xid2], u[uid2], t)

    # linearizing the combined system should be the same as linearizing the
    # subsystems separately
    @test isapprox(lin_dyn.A[xid1, xid1], lin_unicycle1.A)
    @test isapprox(lin_dyn.A[xid2, xid2], lin_unicycle2.A)
    @test isapprox(lin_dyn.B[xid1, uid1], lin_unicycle1.B)
    @test isapprox(lin_dyn.B[xid2, uid2], lin_unicycle2.B)
end
