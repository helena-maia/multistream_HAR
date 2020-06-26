if [ "$#" -ne 4 ]; then
    echo "Usage:" $0 "<modalities> <split> <dataset> <metric>" 
    echo $0 "rgb2,flow,avr 1 ucf101 DFM"
    exit 1
fi

IFS=',' read -ra modalities <<< "$1"
split=$2
dataset=$3
metric=$4 # COR, DFM, QSTAT, IA, DM, COMP
let n=${#modalities[@]}-1

if [ $metric = 'COMP' ]; then
    declare -A matrix
    for i in $(seq 0 $n); do
        matrix[$i,$i]=0
        let i1=$i+1
        for j in $(seq $i1 $n); do
            result=$(python3.6 diversity_measures.py NPYS/ splits/ -s $split -m1 ${modalities[$i]} -m2 ${modalities[$j]} -d $dataset | grep "Comp")
            compAB=$(echo $result | cut -d '=' -f 2 | cut -d ' ' -f 2)
            compBA=$(echo $result | cut -d '=' -f 3)
            matrix[$i,$j]=$compAB
            matrix[$j,$i]=$compBA
        done
    done
    
    printf "\t"
    for i in $(seq 0 $n); do
        printf ${modalities[$i]}"\t"
    done
    echo

    for i in $(seq 0 $n); do
        printf ${modalities[$i]}"\t"
        for j in $(seq 0 $n); do
            printf ${matrix[$i,$j]}"\t"
        done
        echo
    done

else
    printf "MOD1\tMOD2\t$metric\n\n"
    for i in $(seq 0 $n); do
        let i1=$i+1
        for j in $(seq $i1 $n); do
            result=$(python3.6 diversity_measures.py NPYS/ splits/ -s $split -m1 ${modalities[$i]} -m2 ${modalities[$j]} -d $dataset | grep $metric)
            value=$(echo $result | cut -d '=' -f 2)
            printf "${modalities[$i]}\t${modalities[$j]}\t$value\n"
            #printf ${modalities[$i]} "\t" ${modalities[$j]} "\t" $value "\n"
        done
    done
fi
