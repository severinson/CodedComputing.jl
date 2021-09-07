@testset "partition" begin
    for n in 1:17
        for p in 1:n
            Is = [partition(n, p, i) for i in 1:p]
            @test vcat(Is...) == collect(1:n)
            @test maximum(length, Is) <= minimum(length, Is) + 1        
        end
    end
end

@testset "translate_partition" begin
    n = 17
    for p in 1:n
        for i in 1:p
            for q in 1:n
                j = translate_partition(n, p, q, i)
                @test first(partition(n, p, i)) in partition(n, q, j)
            end
        end
    end
end

@testset "align_partitions" begin
    n = 10
    p, q, i = 4, 5, 1
    @test align_partitions(n, p, q, i) == 1
    p, q, i = 4, 5, 2
    @test align_partitions(n, p, q, i) == 2
    p, q, i = 4, 5, 3
    @test align_partitions(n, p, q, i) == 2
    p, q, i = 4, 5, 4
    @test align_partitions(n, p, q, i) == 2
    p, q, i = 4, 2, 2
    @test align_partitions(n, p, q, i) == 1
    p, q, i = 4, 2, 4
    @test align_partitions(n, p, q, i) == 2

    n = 17
    for p in 1:n
        for i in 1:p
            for q in 1:n
                j = align_partitions(n, p, q, i)
                ip = translate_partition(n, q, p, j)
                @test first(partition(n, p, ip)) == first(partition(n, q, j))
            end
        end
    end    
end