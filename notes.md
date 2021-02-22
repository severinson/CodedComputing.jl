## Genome data

### Data format

For PCA, and to plot populations separately, we will use data from the 1000 genomes project. We need two kinds of files: VCF files and PED files.

VCF files record genetic variations, i.e., how genetic samples (e.g., taken from humans, with each sample corresponding to one individual). Each of these possible differences is referred to as a variant. The 1000 genomes VCF files each correspond to one chromosome. Humans have 23 chromosomes, 22 of which are shared by all humans (except in case of certain genetic conditions), and one which corresponds to sex (each human has either a X or Y chromosome).

For each file, each line corresponds to a possible variant, i.e., a possible difference in the sample genome from the reference genome. The first few columns store information about the variant, such as its position in the chromosome, what the reference genome contains at that position, and how the variant may differ. The final columns correspond to genome samples (each sample may correspond to an individual), and the entries of these columns denote how that particular sample differs from the reference (there may be several possibilities).

The PED file stores information about each sample, such as the population it is taken from.

### Workflow

We will compute PCA on a binary matrix, for which each row corresponds to a sample and each column to a position in the genome. For each entry, a 1 indicates that sample differs from the reference (we don't care how it differs).

The strategy used is based on 
https://github.com/bwlewis/1000_genomes_examples
http://bwlewis.github.io/1000_genomes_examples/notes.html
bwlewis.github.io/1000_genomes_examples/PCA.html
bwlewis.github.io/1000_genomes_examples/PCA_overview.html

1. Download a VCF file from the 1000 genomes project for some chromosome ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
2. Convert that file into a CSV file using the parse.c program in the 1000genomes folder
3. Read that CSV file into a Julia DataFrame
4. Create a sparse matrix from the DataFrame by using the second column of the DataFrame as row indices and the first as column indices
5. Compute the principal components of that matrix
6. Project the sample data onto the principal components
7. Match each sample to the population it is taken from
8. Plot the projected samples separately for each population

# Latency variation

Latency when waiting for all workers can vary significantly between experiments. It seems that sometimes you're unlucky and get a slow worker, which increases the overall computation time significantly. I can't do much about this except run more experiments to get a clearer picture of the truth. Data is data.

The intercept still looks like it's well described by a quadratic. I'm less convinced for the slope. It looks like a quadratic when you fix the number of workers, but not globally. It could just be that I don't have enough data to get a clear picture. Let's wait until I have some more data.

# Linear latency model

I need to understand if the linear latency model still works. The intercept part looks fine. The slope less so. It could be that I don't have enough traces or that the traces I do have are biased somehow. When ignoring the data corresponding to nwait = nworkers it looks like a better fit. But still not as good as in the paper. Furthermore, the steepest observed slope is about 0.8, whereas the highest observed slope in the paper is about 2.2, i.e., the slope is only about a third as steep. Meaning that there is even less straggling than what we observed previously.

Let's wait until I have some more traces before continuing with this. It could be an issue with the limited amount of data I have.

The variation between experiments is very large when waiting for all workers. Probably because it's likely that at least 1 worker will be extra slow when waiting for all of them. I should see the same phenomena also for lower nwait with a sufficient number of samples. Further, I should be able to verify that this is indeed the case by looking at the latency of individual workers.

In summary, I want to understand where the large variation between experiments is coming from. My conjecture is that it's due to variation between workers. I could start by looking at it for individual experiments.

Clearly, sometimes you get a good cluster and sometimes you get a bad cluster. Latency isn't always the same. And I'm seeing larger variations than I saw previously because I'm making sure I get a new cluster for every experiment now, which I didn't before. I could simply average across experiments, which is what we did before and what I say we do in the paper. I could also do an average of averages to avoid having experiments with very many iterations influencing the model too much.

This averages of averages across a large number of experiments will likely get me where I want. If it does I have everything I need for this paper in addition to having a natural way to extend it. So let's continue running experiments in the hope that it sorts out my model.

Another way to recover the slope that might be less noisy is to look at the latency of individual workers. I could look at how much latency increases when going from the fastest to slowest worker. If that relationship is linear it would give me a way to recover the intercept and slope. It would also mean I wouldn't have to mess around with the Gamma variables. It's just mean response time of each worker in a given job sorted from lowest to highest. If latency does actually increase linearly with w, then I should see that behavior in the per-worker latency too.

I do see a linear increase in latency as a function of w when looking at individual traces.

It checks out. And it's beautiful. Looks rather linear. But not quite. It sort of tapers at the edges around nwait=1 and nwait=nworkers, which is strange.

I don't see any correlation between subpartition density and latency.

Let's work on a linear latency model based on the order stats samples. Let's start by capturing the latency of the fastest worker. I want the latency of the fastest worker as a function of nworkers and worker_flops. Let's plot the latency of the fastest worker as a function of worker_flops for al nworkers. I would expect that it's an increasing function of the flops and decreasing with the number of workers.

It looks like the latency is best described by a degree 3 polynomial, which is problematic since it becomes much harder to reason about the parameters. Except for the intercept.

I don't know what to do. The goal is to model the latency of the w-th fastest worker in a way that lets me predict it. For a given nworkers, w, and c it looks like a degree-3 polynomial. The fit is very accurate, which is great. Problem is it makes it hard to interpolate those parameters. Before I used a degree-1 polynomial, which made reasoning about the parameters very easy. I need to somehow reduce the degrees of freedom. It makes sense that the intercept is invariant of the number of workers. It's just the minimum expected amount of time needed to do the task. Everything after that should depend on the number of workers. Further, there seems to be some symmetry. The curve is convex for the first half and concave for the second half. It has an inflection point in the middle.

Found something quite extraordinary. The latency difference due to waiting for more workers looks like a quadratic. And it looks to be the same for all scenarios with equal load. It makes sense that the intercept is the same regardless of the number of workers. The difference being the same means that the extra latency of waiting for 1 more worker when there are 18 workers in total is the same as the extra latency of waiting for two more workers when the there are 36 workers in total. 

Something strange going on with order stats latency. Latency of a particular order increases with w. I think because some workers will still be straggling when the next iteration starts. My implementation may be amplifying this behavior. Since I first give work to the non-stragglers of the previous iteration before seeing if the stragglers from the previous iteration are done. This is probably the way to do it to maximize speed. I want my implementation to have as small an effect as possible though.

Right, I've updated MPIStragglers.jl to process stragglers from previous iterations first. That should reduce the correlation between iterations. Let's see how much of a difference it makes. Turns out the difference is small.

This non-iid-ness is interesting. Because it tells you that a latency model fit to only a single iteration will be off since a given worker may still be unavailable in subsequent iterations. But it doesn't make sense that w=1 is between w=18 and w=36. However, I'm not controlling for the time between iterations, which is a function of w.

I want to capture the expected latency of the w-th fastest worker out of N for a given computational load. In the end, what I want is to capture iteration latency.

However, the latency of the w-th fastest worker isn't independent of w_target. I think because w_target determines how many workers are available during the next iteration.

I've no idea what's going on. The results I'm seeing aren't making much sense. I'm not in control of all of the variables, which is an issue. Let's write a new kernel where I can control everything.
- Number of workers
- Number of workers to wait for
- Computational workload (rows, columns, sparsity)
- Time between iterations
- Amount of communication

I've written the latency kernel. Let's run it. And let's run it for settings similar to those used for the PCA kernel. But let's do it for both the original and transposed file. Experiments are running. Time to write some code for analysis and plotting.

Looking at the data from the latency kernel things look linear again. What a rollercoaster.
It seems that straggling is linear and makes sense when computation latency is the dominating factor. Things become weird when communication dominates.
So maybe the model we've been looking at is fine. It just doesn't work when latency is due to communication.


# Gamma latency model

Let's work on the latency of individual workers in the meantime. It looks like the response time of individual workers is Gamma. That is, the latency of an individual worker is described by a Gamma random variable with some parameters. I want to understand what the spread is for those those parameters. I need to have a rather large number of samples from each individual worker in order to do this.

I've made a first pass over the data. It seems like the mean increases linearly with the computational load. This makes sense. Higher load should mean higher latency. The impact of increased computational load on the is less clear-cut. It goes down with increased load. The scale also goes down with increased load. A possible explanation is that tail is caused by random delays, which make up a smaller fraction of the delay for longer jobs.

There doesn't seem to by any correlation between the mean and scale.

Capturing the distribution of the parameters of the Gamma would result in a very nice and sophisticated latency model. However, it may be best left for a future paper. First, let's focus on the current paper. I need to know if the linear model is garbage or not.

# New scenario

I think I have a way to get a scenario where we do better: compute the left-singular vectors of the entire genome dataset. From the left-singular vectors one can easily compute the right singular vectors using 1 matrix-matrix multiplication. This also reduces the amount of communication and the amount of work done by the coordinator. It means I can tune the load at the workers without changing the load at the coordinator. However, I should make sure to first understand what's going on for the dataset I already have.

Computing the left-singular vectors makes a ton more sense. Makes everything simpler. So let's run some experiments for this scenario. I do need to put in some more work for that though. But if these things work out, I'm well on my way to fixing the issues I've been having.

# 210215

- I've re-created the important plots from the paper using the new data, which is based on experiments with shuffled data
- It looks like DSAG can still have an advantage over SAG
- There's still a lot of correlation
- However, the model doesn't look so good anymore
- Let's compare latency recorded individually for different nwait

Meeting prep.
- Explain the mistake and why it's important
- It would be difficult for a reviewer to notice this mistake
- Show them how it affects our current plots
- There's still straggling and stragglers still remain stragglers, but we've got to update the model to reflect what we see
- To do that I've recorded latency individually for each worker for SGD
- Show them latency recorded individually per worker
- There's a discrepancy, which I'm investigating
- The data suggests that latency is better modelled by a degree 3 polynomial
- Show them the derivative, which suggests a new latency model
- It still seems like the derivative wrt w is is given by the load and the number of workers
- We may be able to propose a more accurate model
- I've also written a new kernel explicitly for measuring latency that gives more control over all parameters
- Regarding scenario, we can rewrite the PCA problem to put less load on the coordinator and to require less communication
- The data we have indicates that this will give DSAG a larger edge


- I've cleaned up several things today
- Now I'm waiting for latency data that'll hopefully make it more clear how latency works on aws
- I'm currently waiting for experiments for 18 workers to finish
- The hope is that this data will make more sense than what I currently have and that I'll be able to use it to develop a more accurate model
- If I can create a better model I can figure out where straggler resiliency is important
- It'll probably take a while for me to get the data I need
- I also need to run experiments for some other number of workers

# 210217

- The idea is to model each worker by a random variable
- Now I want to sample from that distribution
- Let's start by looking at the latency distribution of individual workers
- I'll plot the distribution of some randomly selected individual worker together with fitted distributions
- It looks like the shifted Gamma is the best fit. At least for the lower quantiles we're interested in (since we're trying to capture the mean)
- However, it also seems like workers exist in one of two states, slow and fast
- Slow iterations are typically clustered
- I've captured the mean and variance of the worker compute latency as a function of the computational load
- I did it as two 

Modeling procedure
- For each worker, draw the mean and variance of that worker's latency from a given probability distribution (the normal distribution)

- I'm trying to model the process that creates workers
- I'm assuming that, for a given computational load, each worker is characterized by the mean, variance, and minimum of its compute latency
- That means I need to capture the distribution of those parameters
- It looks like the mean and minimum can be captured by a 2-dimensional Normal distribution
- It's less clear for the variance
- By looking at the worker latency time-series I see that workers have some particular baseline latency but experience bursts of high latency
- For now I just want to model the baseline latency. Hence, I need some way to remove these bursts
- It's a bi-modal distribution, i.e., it has two humps
- I want to model those two humps separately, or at least the first hump separately
- I could try to isolate the two humps by looking at the derivative of the pdf, which will be zero somewhere between the two humps
- But if the distribution is the sum of two humps they will overlap
- That means I need to cut out those samples to isolate them
- I need some method for cutting out those samples
- Something is causing these latency spikes and I have no idea what it is
- I need to make a decision on how much effort I should put into figuring out what it is
- I could try to figure it out
- On the other hand, the point of what I'm doing is to build systems resilient to this kind of noise, i.e., to make it so that one doesn't need to understand the cause
- So let's not do it for now
- Instead, I'll try to analyze the data I have
- Say that latency is determined by minimum latency + fast Gamma noise process + slow noise process
- It looks like I can capture the minimum latency and the fast Gamma process
- Now I need to consider the slow process
- The slow process obviously has some memory
- Let's fix the computational load and compute the mean over 100 iteration
- 0.0397 seems like a good limit for a two-state process for worker_flops=2.52e7
- I have two options to model latency
- Either it's the sum of two stochastic process, one fast memoryless and one slow with memory
- Or latency is the realization of a random variable the parameters of which are stochastic processes
- For the first I need to determine the number of states and so on
- The second option doesn't have this limitation

# 210216

The goal for today is to develop a latency model
The model should predict the expected latency of the w-th fastest worker
The problem I had before was that I saw different latency for the w-th fastest worker depending on what w_target was
So I first need to check if that problem has gone away after the changes I've made

I'm considering two different scenarios
The computational load is the same for both, but they differ by the amount of aommunication
The first scenario requires a factor 724 more communication than the second
For the first scenario there is a significant difference between the iteration latency and the latency of individual workers
There's also a significant difference 
(the iteration latency is higher)
The only explanation I can come up with is that the difference is due to the time needed for the coordinator to store all the additional data
The second scenario is much faster. And there's much less straggling
Hence, straggling is mostly caused by communication
I need a scenario where straggling is a major problem
Apparently that means a scenario with lots of communication
For now, let's focus on understanding and modeling

Let's consider the following
- How latency is affected by computational load
- How latency is affected by the amount of communication
- How latency is affected by the number of workers

I actually record the compute latency separately
Plotting the compute latency indicates that there's straggling going on even when only doing compute
The latency as a function of w for compute latency looks like a degree 3 polynomial
Let's see if I can capture the parameters of that polynomial

The difference in latency due to waiting for one more worker looks constant except for at the edges, around w=1 and w=nworkers
Not quite constant though
Some workers are exceptionally slow and some and exceptionally fast
And it looks quite symmetric, i.e., the number of fast workers is about the same as the number of slow workers
What about if I looked at the average compute latency of individual workers for a given computational load?

It looks like the compute latency of individual workers follows a Normal distribution
The fit becomes better when workers do less work
The normal distribution is parameterized by the mean and variance
Let's capture the mean and variance as a function of worker_flops

Feels like it's starting to be time to write some stuff down

- The latency of each worker at a partiular point of time is characterized by a random variable with some distribution
- The parameters of that distribution are in turn drawn from some other distribution
- The parameters of that distribution may change over time, which is captured by the autocorrelation of those parameters
- Latency may be correlated between workers
- The latency of a particular iteration is equal to the latency of the w-th fastest worker
- Which is captured by the w-th order statistic of the random variables characterizing the latency of the individual workers
- To sample from the iteration latency distribution I need to first generate a set of distributions corresponding to the workers, then draw a sample from each of those distributions, and finally take the w-th largest value of out those samples

# 210218

- I used to think that each worker was characterized by a probability distribution with parameters drawn from some other probability distribution
- It seems that it's more accurate to characterize workers by a stochastic process
- Tukey-Lambda with lambda=-0.4 seems like a good fit
- That means heavier tails than Normal, but not as heavy as Cauchy
- I need the minimum latency and the scale and shape of the noise process
- I think I can get the shape from the difference between any two quantiles
- I have a model for latency. Now it's time to see how good it is
  - Minimum latency drawn from a Normal distribution
  - Shape parameter 0.4
- Let's model the latency as the sum of several processes of different speed
- Constant minimum latency, fast noise, some number of slow processes
- I need to isolate the different processes
- Let's just go from slowest to fastest
- It anyway makes sense that latency is the sum of different processes operating at different time scales
- Then I just need to identify the number of processes active and isolate them
- FFT would be useful for that
- Let's just figure out some heuristic for finding what state each worker is in
- There are two independent random processes
- I need to determine which state the Markov chain is in
- I want an automatic way to mark which samples correspond to which underlying Markov state

# 210219

- I've managed to isolate the large latency spikes
- Next challenge is to model the remaining latency
- There's definitely correlation between adjacent samples, i.e., it's a random process with non-zero autocorrelation
- Latency is surely additive
- I just need to isolate the different behaviors going on
- The overall latency is made up of the sum of
  - A constant
  - A Markov process
  - Random iid noise
  - Extra latency if the worker is currently experiencing a burst
- Let's discretize the Markov process and compute transition probabilities
- I've computed the state transition matrix for the Markov chain
- I can use it to generate realizations of the stochastic process
- Latency is the sum of a constant, a markov chain, iid noise, and bursts
- Now I just need to model all of these things

# 210222

- Now I need to isolate the different components of the latency
- First, the distribution of the mean of the latency over 300s (let's call it the constant part)
- Constant part seems to be a Normal distribution across workers and jobs
- Now let's do the high-frequency i.i.d. noise part
- High-frequency noise looks like a Normal distribution
- So let's look at the distribution of the parameters of that normal
- Mean looks like something between a Normal and a Cauchy. Maybe a shape parameter somewhere between 0 and -3.
- Variance looks like a Gamma
- Next up is the Markov process, which is characterized by the states and the transition probability matrix
- I can fix the states
- I need to do something smart about the transition probability matrix
- Maybe I can parameterize it somehow?
- Let's ignore the Markov for now
- Instead, let's model the mean, noise, and bursts only and see how close to the truth that gets us
- Latency during bursts looks basically Normal
- Next is to compute the state transition matrix, i.e., the prob. of being in a burst