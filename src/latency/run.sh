jobfile=./src/latency/latency.job
while true
do
    
    # make sure there is disk space available
    if [ $(df --output=pcent /shared | tr -dc '0-9') -gt 90 ]
    then
        echo "Disk is more than 90% full. Waiting until the disk is at most 70% full..."
        while [ $(df --output=pcent /shared | tr -dc '0-9') -gt 70 ]
        do
            sleep 10
        done
    fi

    # wait for current jobs to finish
    echo "Waiting for all jobs to finish..."
    while [ $(squeue | wc -l) -gt 1 ]
    do
        sleep 10
    done

    # wait until servers to become idle so that we get new ones for each sbatch call
    echo "Waiting for workers to become idle..."
    sleep 15m

    # submit new job
    echo "Submitting job."
    sbatch ${jobfile}
    sleep 10
done