kernel = "../src/pca/logreg.jl" 

function logreg_loss(v, X, b, λ)
    rv = 0.0
    for i in 1:length(b)
        rv += log(1 + exp(-b[i]*(v[1]+dot(X[:, i], view(v, 2:length(v))))))
    end
    rv / length(b) + λ/2 * norm(v)^2
end

# input data set with known optimal solution
X = [1.444786643000158 0.49236792885913283 -0.53258473265429 0.05476455630673194 -1.3473893605265843; 0.48932299731783646 2.0708445447107926 1.2414596020757043 0.9131934117095984 -0.15692043560721075; 0.7774625331093794 0.7234405608945721 -0.037446104354257874 -1.1104987697394342 1.354975413199728]
b = [-1, 1, -1, -1, 1]
v_opt = [-0.3423591553493419, -0.41317049965033387, 0.007294166575956451, 0.6763846515861628]
m, n = size(X)
λ = 1 / n
opt = logreg_loss(v_opt, X, b, λ)

# write test problem to disk
inputfile = tempname()
inputdataset = "X"
outputdataset = "V"
labeldataset = "b"
h5open(inputfile, "w") do file
    file[inputdataset] = X
    file[labeldataset] = b
end

# GD
nworkers = 2
niterations = 200
stepsize = 0.1
outputfile = tempname()
mpiexec(cmd -> run(```
    $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
    --inputdataset $inputdataset
    --outputdataset $outputdataset
    --niterations $niterations
    --saveiterates
    --lambda $λ
    ```))
vs = load_logreg_iterates(outputfile, outputdataset)
v = vs[end]
f = logreg_loss(v, X, b, λ)
@test f < opt * (1+1e-2)

# same as previous, but with nslow > 0
nworkers = 2
nslow = 1
niterations = 100
stepsize = 0.1
outputfile = tempname()
mpiexec(cmd -> run(```
    $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
    --inputdataset $inputdataset
    --outputdataset $outputdataset
    --niterations $niterations
    --saveiterates
    --lambda $λ
    --nslow $nslow
    ```))
vs = load_logreg_iterates(outputfile, outputdataset)
v = vs[end]
f = logreg_loss(v, X, b, λ)
@test f < opt * (1+1e-2)

# same as previous, but with slowprob. > 0
nworkers = 2
slowprob = 0.5
niterations = 100
stepsize = 0.1
outputfile = tempname()
mpiexec(cmd -> run(```
    $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
    --inputdataset $inputdataset
    --outputdataset $outputdataset
    --niterations $niterations
    --saveiterates
    --lambda $λ
    --slowprob $slowprob
    ```))
vs = load_logreg_iterates(outputfile, outputdataset)
v = vs[end]
f = logreg_loss(v, X, b, λ)
@test f < opt * (1+1e-2)

# DSAG
vralgo = "tree"
nworkers = 2
nwait = 1
niterations = 100
stepsize = 0.1
nsubpartitions = 2
outputfile = tempname()
mpiexec(cmd -> run(```
    $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
    --inputdataset $inputdataset
    --nwait $nwait
    --variancereduced
    --vralgo $vralgo
    --nsubpartitions $nsubpartitions
    --outputdataset $outputdataset
    --niterations $niterations
    --saveiterates
    --lambda $λ
    ```))
vs = load_logreg_iterates(outputfile, outputdataset)
v = vs[end]
f = logreg_loss(v, X, b, λ)    
@test f < opt * (1+1e-2)

# DSAG w. nwaitschedule < 1.0
nworkers = 2
nwait = 2
niterations = 100
stepsize = 0.1
nsubpartitions = 2
nwaitschedule = 0.9
outputfile = tempname()
mpiexec(cmd -> run(```
    $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
    --inputdataset $inputdataset
    --nwait $nwait
    --variancereduced
    --vralgo $vralgo    
    --nsubpartitions $nsubpartitions
    --outputdataset $outputdataset
    --niterations $niterations
    --saveiterates
    --lambda $λ
    --nwaitschedule $nwaitschedule
    ```))
vs = load_logreg_iterates(outputfile, outputdataset)
v = vs[end]
f = logreg_loss(v, X, b, λ)    
@test f < opt * (1+1e-2)

# DSAG w. sparse input data
inputfile = tempname()
h5open(inputfile, "cw") do file
    H5SparseMatrixCSC(file, inputdataset, sparse(X))
    file[labeldataset] = b
    flush(file)
end

nworkers = 2
nwait = 1
niterations = 100
stepsize = 0.1
nsubpartitions = 2
outputfile = tempname()
mpiexec(cmd -> run(```
    $cmd -n $(nworkers+1) julia --project $kernel $inputfile $outputfile        
    --inputdataset $inputdataset
    --nwait $nwait
    --variancereduced
    --vralgo $vralgo    
    --nsubpartitions $nsubpartitions
    --outputdataset $outputdataset
    --niterations $niterations
    --lambda $λ
    ```))
vs = load_logreg_iterates(outputfile, outputdataset)
v = vs[end]
f = logreg_loss(v, X, b, λ)
@test f < opt * (1+1e-2)