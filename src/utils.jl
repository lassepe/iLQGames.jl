macro S(gen)
    :(SVector{length($gen)}($gen))
end
