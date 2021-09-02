function tg_sum(tg::TreeGradient)
    rv = zero(tg.∇)
    for (_, node) in tg
        rv .+= node.∇
    end
    rv
end

function tg_cmp(a, b)
    key_a, node_a = a
    key_b, node_b = b
    @test key_a == key_b
    @test node_a.∇ ≈ node_b.∇
    @test node_a.isinitialized == node_b.isinitialized
    return
end

∇ = fill(1.0, 2)
n = 10
tg = TreeGradient(∇, n)
ans = collect(tg)
@test length(ans) == 1
tg_cmp(ans[1], (1, TreeGradientNode(fill(1.0, 2), true)))
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 10

# make a series of insertions, and test that the state is as expected
i, j = 2, 8
v = 2.0
∇i = fill(v, 2)
insert!(tg, i, j, ∇i)
∇i .= 0
ans = collect(tg)
@test length(ans) == 3
tg_cmp(ans[1], (1, TreeGradientNode(zeros(2), false)))
tg_cmp(ans[2], (2, TreeGradientNode(fill(v, 2), true)))
tg_cmp(ans[3], (9, TreeGradientNode(zeros(2), false)))
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 7

i, j = 2, 8
v = 2.1
∇i = fill(v, 2)
insert!(tg, i, j, ∇i)
∇i .= 0
ans = collect(tg)
@test length(ans) == 3
tg_cmp(ans[1], (1, TreeGradientNode(zeros(2), false)))
tg_cmp(ans[2], (2, TreeGradientNode(fill(v, 2), true)))
tg_cmp(ans[3], (9, TreeGradientNode(zeros(2), false)))
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 7

i, j = 2, 10
v = 3.0
∇i = fill(v, 2)
insert!(tg, i, j, ∇i)    
∇i .= 0
ans = collect(tg)
@test length(ans) == 2
tg_cmp(ans[1], (1, TreeGradientNode(zeros(2), false)))
tg_cmp(ans[2], (2, TreeGradientNode(fill(v, 2), true)))
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 9

i, j = 4, 6
v = 4.0
∇i = fill(v, 2)
insert!(tg, i, j, ∇i)
∇i .= 0
ans = collect(tg)
@test length(ans) == 4
tg_cmp(ans[1], (1, TreeGradientNode(zeros(2), false)))
tg_cmp(ans[2], (2, TreeGradientNode(zeros(2), false)))
tg_cmp(ans[3], (4, TreeGradientNode(fill(v, 2), true)))
tg_cmp(ans[4], (7, TreeGradientNode(zeros(2), false)))
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 3

i, j = 1, 10
v = 5.0
∇i = fill(v, 2)
insert!(tg, i, j, ∇i)
∇i .= 0
ans = collect(tg)
@test length(ans) == 1
tg_cmp(ans[1], (1, TreeGradientNode(fill(v, 2), true)))
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 10

i, j = 1, 3
∇i = fill(1.0, 2)
insert!(tg, i, j, ∇i)
∇i .= 0
i, j = 4, 4
∇i = fill(3.0, 2)    
insert!(tg, i, j, ∇i)
∇i .= 0
i, j = 5, 6
∇i = fill(4.0, 2)    
insert!(tg, i, j, ∇i)
∇i .= 0
i, j = 2, 8
∇i = fill(8.0, 2)    
insert!(tg, i, j, ∇i)
∇i .= 0
ans = collect(tg)
@test length(ans) == 3
tg_cmp(ans[1], (1, TreeGradientNode(zeros(2), false)))
tg_cmp(ans[2], (2, TreeGradientNode(fill(8.0, 2), true)))
tg_cmp(ans[3], (9, TreeGradientNode(zeros(2), false)))
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 7

i, j = 1, 1
∇j = fill(2.0, 2)
insert!(tg, i, j, ∇j)
∇j .= 0
ans = collect(tg)
@test length(ans) == 3
tg_cmp(ans[1], (1, TreeGradientNode(fill(2.0, 2), true)))
tg_cmp(ans[2], (2, TreeGradientNode(fill(8.0, 2), true)))
tg_cmp(ans[3], (9, TreeGradientNode(zeros(2), false)))
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 8

i, j = 2, 10
∇i = fill(3.0, 2)
insert!(tg, i, j, ∇i)
∇i .= 0
ans = collect(tg)
@test length(ans) == 2
tg_cmp(ans[1], (1, TreeGradientNode(fill(2.0, 2), true)))
tg_cmp(ans[2], (2, TreeGradientNode(fill(3.0, 2), true)))
@test tg.∇ ≈ tg_sum(tg)
@test tg.ninit == 10

# test that new memory was allocated
∇j .= 1.0
tg_cmp(ans[1], (1, TreeGradientNode(fill(2.0, 2), true)))

# test input validation
i, j = 0, 3
@test_throws ArgumentError insert!(tg, i, j, ∇i)
i, j = 1, n+1
@test_throws ArgumentError insert!(tg, i, j, ∇i)
i, j = 1, 2
∇i = zeros(3)
@test_throws DimensionMismatch insert!(tg, i, j, ∇i)