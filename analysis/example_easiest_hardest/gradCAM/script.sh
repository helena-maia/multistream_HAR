while IFS=',' read -r -a array; do
    #echo ${array[2]}

    d=${array[6]}

    if [ ${array[5]} == "RGB*" ]; then
    	python spatial_demo.py -m rgb -d $d -s 1 -a inception_v3 --settings ../settings_gradCAM ${array[2]} ${array[3]}
    elif [ ${array[5]} == "AVR" ]; then
    	python spatial_demo.py -m rhythm -d $d -s 1 -a inception_v3 -vra 3 --settings ../settings_gradCAM ${array[2]} ${array[3]}
    elif [ ${array[5]} == "FLOW" ]; then
    	python temporal_demo.py -m flow -d $d -s 1 -a inception_v3 --settings ../settings_gradCAM ${array[2]} ${array[3]}
    else
    	python rhythm.py -d $d -s 1 -a inception_v3 --settings ../settings_gradCAM ${array[2]} ${array[3]}
    fi

    #python spatial_demo.py -m rgb2 -d ucf101 -s 1 -a inception_v3 -vra 3 --settings ../settings_gradCAM /home/Datasets/ /log
done < npy_log_list.csv