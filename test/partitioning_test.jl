for n in 1:17
    for p in 1:n
        Is = [partition(n, p, i) for i in 1:p]
        @test vcat(Is...) == collect(1:n)
        @test maximum(length, Is) <= minimum(length, Is) + 1        
    end
end