"A non-allocating versin of norm(x::SVector, Inf)"
@inline infnorm(x::StaticVector) = maximum(abs.(x))
