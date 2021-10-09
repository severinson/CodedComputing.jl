@testset "Basic tests" begin
    vs = 1:100
    n = 10    
    ch = ConcurrentCircularBuffer{Int}(n)

    # pop
    append!(ch, vs)
    rv = zeros(Int, n)
    for i in 1:n
        v = pop!(ch)
        rv[i] = v
    end
    @test rv == reverse(vs[end-n+1:end])

    # pop several
    append!(ch, vs)    
    rv = zeros(Int, n)
    i = pop!(rv, ch)
    @test i == n
    @test rv == reverse(vs[end-n+1:end])

    # popfirst
    append!(ch, vs)
    rv = zeros(Int, n)
    for i in 1:n
        v = popfirst!(ch)
        rv[i] = v
    end
    @test rv == vs[end-n+1:end]

    # popfirst several
    append!(ch, vs)    
    rv = zeros(Int, n)
    i = popfirst!(rv, ch)
    @test i == n
    @test rv == vs[end-n+1:end]
end

@testset "Threading" begin
    n = 1000

    # one producer and one consumer
    ch = ConcurrentCircularBuffer{Int}(n)    
    rv = zeros(Int, n)
    consumer = Threads.@spawn begin
        try
            i = 1
            while true
                rv[i] = popfirst!(ch)
                i += 1
            end
        catch e
            if e isa InvalidStateException
            else
                rethrow()
            end
        end
    end
    producer = Threads.@spawn for i in 1:n
        push!(ch, i)
    end
    wait(producer)
    close(ch)
    wait(consumer)
    @test rv == 1:n

    # one producer and two consumers
    ch = ConcurrentCircularBuffer{Int}(n)    
    rv1 = zeros(Int, n)
    ts1 = zeros(n)
    consumer1 = Threads.@spawn begin
        try
            for i in 1:n
                ts1[i] = @elapsed rv1[i] = popfirst!(ch)
            end
        catch e
            if e isa InvalidStateException
            else
                rethrow()
            end
        end
    end
    rv2 = zeros(Int, n)
    ts2 = zeros(n)
    consumer2 = Threads.@spawn begin
        try
            for i in 1:n
                ts2[i] = @elapsed rv2[i] = popfirst!(ch)
            end
        catch e
            if e isa InvalidStateException
            else
                rethrow()
            end
        end
    end
    tsc = zeros(4n)
    producer = Threads.@spawn for i in 1:4n
        tsc[i] = @elapsed push!(ch, i)
    end
    wait(producer)
    close(ch)
    wait(consumer1)
    wait(consumer2)
    @test issorted(rv1)
    @test issorted(rv2)
    @test length(intersect(Set(rv1), Set(rv2))) == 0
    @info "consumer1 throughput: $(1 / (sum(ts1) / n)), consumer2 throughput: $(1 / (sum(ts2) / n))"
end