struct AffineStrategy{nx, nu, TP<:SMatrix{nu, nx}, TA<:SVector{nu}}
    "The proportional gain."
    P::TP
    "The feed-forward term."
    α::TA
end

n_states(γ::AffineStrategy{nx}) where {nx} = nx
n_controls(γ::AffineStrategy{nx, nu}) where {nx, nu} = nu

Base.zero(γ::AffineStrategy) = zero(typeof(γ))
function Base.zero(::Type{<:AffineStrategy{nx, nu, TP, TA}}) where {nx, nu, TP, TA}
    return AffineStrategy(zero(TP), zero(TA))
end

"""
    $(TYPEDSIGNATURES)

Computes the control input for a given affine strategy γ, for a state reference
deviation of Δx and a control reference of ũ.
"""
control_input(γ::AffineStrategy, Δx::AbstractVector, ũ::AbstractVector) = ũ - γ.P*Δx - γ.α
