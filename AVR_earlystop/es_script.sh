DELTA="0.00 0.05 0.10 0.15 0.20"
PATIENCE="1 2 3 4 5 6 7"

echo path timestamp dataset modality split delta patience epoch train_prec val_prec

while read line;  do 
    data=($line)
    path=${data[0]}
    timestamp=${data[1]}
    dataset=${data[2]}
    modality=${data[3]}
    s=${data[4]}

    for d in $DELTA; do
        for p in $PATIENCE; do
            ret=$(python es_precision_dir.py $path/$timestamp/early_stopping.json $path/$timestamp/precision $d $p | tail -n 1)
            echo $line $d $p $ret
        done
    done


done < es_list.csv
