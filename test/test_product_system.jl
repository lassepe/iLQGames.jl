using Test

using iLQGames:
    ProductSystem,
    Car5D,
    n_states,
    n_controls,
    dx,
    linearize

using StaticArrays


# create two separate dynamics systems
car1 = Car5D(1.0)
car2 = Car5D(1.0)
xid1 = SVector{5}(1:5)
xid2 = SVector{5}(6:10)
uid1 = SVector{2}(1:2)
uid2 = SVector{2}(3:4)

# and combine them in the product system
dyn = ProductSystem((car1, car2))

# test some operations on the joint system

x = @SVector rand(n_states(dyn))
u = @SVector rand(n_controls(dyn))
t = 0.0;

@testset "dx" begin
    dx_dyn = dx(dyn, x, u, t)
    dx_car1 = dx(car1, x[xid1], u[uid1], t)
    dx_car2 = dx(car2, x[xid2], u[uid2], t)

    @test isapprox(dx_dyn[xid1], dx_car1)
    @test isapprox(dx_dyn[xid2], dx_car2)
end

@testset "linearize" begin
    lin_dyn = linearize(dyn, x, u, t)
    lin_car1 = linearize(car1, x[xid1], u[uid1], t)
    lin_car2 = linearize(car2, x[xid2], u[uid2], t)

    # linearizing the combined system should be the same as linearizing the
    # subsystems separately
    @test isapprox(lin_dyn.A[xid1, xid1], lin_car1.A)
    @test isapprox(lin_dyn.A[xid2, xid2], lin_car2.A)
    @test isapprox(lin_dyn.B[xid1, uid1], lin_car1.B)
    @test isapprox(lin_dyn.B[xid2, uid2], lin_car2.B)
end
