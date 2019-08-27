@generated function static_splat(f::Function, v::Union{SVector{S, T}, MVector{S, T}}) where {S, T}
    expr = Expr(:call, :f)
    for i in 1:S
        push!(expr.args, :(v[$i]))
    end
    return expr
end
