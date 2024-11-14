for m in "llama3.1:70b"
do
    for s in "mmr" "similarity"
    do
        for n in 1 5 10 500 1000
        do
            echo "Running $m ($s) using $n results from powertrain"
            python run-experiment.py -s 149929 -e 150029 --search_strategy $s --num_of_results $n --model $m --collection powertrain
        done
    done
done