function tg_sum(tg::TreeGradient)
    rv = zero(tg.∇)
    for (_, v) in tg
        rv .+= v
    end
    rv
end

∇ = fill(1.0, 2)
n = 10
tg = TreeGradient(∇, n)
@test collect(tg) == [(1, fill(1.0, 2))]
@test tg.∇ ≈ tg_sum(tg)

# make a series of insertions, and test that the state is as expected
i, j = 2, 8
∇i = fill(2.0, 2)
insert!(tg, i, j, ∇i)
@test collect(tg) == [(1, zeros(2)), (2, ∇i), (9, zeros(2))]
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 7

i, j = 2, 8
∇i = fill(2.1, 2)
insert!(tg, i, j, ∇i)    
@test collect(tg) == [(1, zeros(2)), (2, ∇i), (9, zeros(2))]
@test tg.∇ ≈ tg_sum(tg)

i, j = 2, 10
∇i = fill(3.0, 2)
insert!(tg, i, j, ∇i)    
@test collect(tg) == [(1, zeros(2)), (2, ∇i)]
@test tg.∇ ≈ tg_sum(tg)

i, j = 4, 6
∇i = fill(4.0, 2)
insert!(tg, i, j, ∇i)    
@test collect(tg) == [(1, zeros(2)), (2, zeros(2)), (4, ∇i), (7, zeros(2))]
@test tg.∇ ≈ tg_sum(tg)

i, j = 1, 10
∇i = fill(5.0, 2)
insert!(tg, i, j, ∇i)    
@test collect(tg) == [(1, ∇i)]
@test tg.∇ ≈ tg_sum(tg)

i, j = 1, 3
∇i = fill(1.0, 2)
insert!(tg, i, j, ∇i)    
i, j = 4, 4
∇i = fill(3.0, 2)    
insert!(tg, i, j, ∇i)
i, j = 5, 6
∇i = fill(4.0, 2)    
insert!(tg, i, j, ∇i)
i, j = 2, 8
∇i = fill(8.0, 2)    
insert!(tg, i, j, ∇i)
@test collect(tg) == [(1, zeros(2)), (2, fill(8.0, 2)), (9, zeros(2))]
@test tg.∇ ≈ tg_sum(tg)

# test that new memory was allocated
∇i .= 1.0
@test collect(tg) == [(1, zeros(2)), (2, fill(8.0, 2)), (9, zeros(2))]

# test input validation
i, j = 0, 3
@test_throws ArgumentError insert!(tg, i, j, ∇i)
i, j = 1, n+1
@test_throws ArgumentError insert!(tg, i, j, ∇i)
i, j = 1, 2
∇i = zeros(3)
@test_throws DimensionMismatch insert!(tg, i, j, ∇i)