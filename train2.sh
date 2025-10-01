#!/bin/bash
device=0
epoch1=15
epoch2=10
times=17

obj_list=("carrot" "dowel" "cookie")
cls_ids=(0 1 2)
for cls_id in "${!cls_ids[@]}";do
    LOG=${save_dir}"res.log"
    echo ${LOG}
    echo ${cls_id} 
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_mvtec_3d
            save_dir=./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]}/weights/

            CUDA_VISIBLE_DEVICES=${device} python train.py --dataset mvtec_pc_3d_rgb \
            --train_data_path /data1/dengzehao/new/data/mvtec_3d \
            --save_path ${save_dir}stage1 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch1} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.002 \
            --train_dataset_name ${obj_list[cls_id]} \
            

            CUDA_VISIBLE_DEVICES=${device} python train2.py --dataset mvtec_pc_3d_rgb \
            --train_data_path /data1/dengzehao/new/data/mvtec_3d \
            --save_path ${save_dir}stage2 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch2} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.0005 \
            --train_dataset_name ${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth
            

            CUDA_VISIBLE_DEVICES=${device} python test2.py --dataset mvtec_pc_3d_rgb \
            --data_path /data1/dengzehao/new/data/mvtec_3d \
            --save_path ./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth \
            --stage2_checkpoint_path ${save_dir}stage2/epoch_${epoch2}.pth \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --train_class ${obj_list[cls_id]}
        wait
        done
    done
done



obj_list=('shell' 'starfish' 'seahorse')
cls_ids=(0 1 2) 
for cls_id in "${!cls_ids[@]}";do
    LOG=${save_dir}"res.log"
    echo ${LOG}
    echo ${cls_id} 
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_real_3d
            save_dir=./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]}/weights/

            CUDA_VISIBLE_DEVICES=${device} python train.py --dataset real_pc_3d_rgb \
            --train_data_path /data1/dengzehao/new/data/Real3D-AD \
            --save_path ${save_dir}stage1 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch1} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.002 \
            --train_dataset_name ${obj_list[cls_id]} \
            

            CUDA_VISIBLE_DEVICES=${device} python train2.py --dataset real_pc_3d_rgb \
            --train_data_path /data1/dengzehao/new/data/Real3D-AD \
            --save_path ${save_dir}stage2 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch2} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.0001 \
            --train_dataset_name ${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth
            

            CUDA_VISIBLE_DEVICES=${device} python test_only_point2.py --dataset real_pc_3d_rgb \
            --data_path /data1/dengzehao/new/data/Real3D-AD \
            --save_path ./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth \
            --stage2_checkpoint_path ${save_dir}stage2/epoch_${epoch2}.pth \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --train_class ${obj_list[cls_id]}
        wait
        done
    done
done



obj_list=('PeppermintCandy' 'LicoriceSandwich' 'Confetto')
cls_ids=(0 1 2)
for cls_id in "${!cls_ids[@]}";do
    LOG=${save_dir}"res.log"
    echo ${LOG}
    echo ${cls_id} 
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_eye_3d
            save_dir=./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]}/weights/

            CUDA_VISIBLE_DEVICES=${device} python train.py --dataset eye_pc_3d_rgb \
            --train_data_path /data1/dengzehao/new/data/Eyecandies_processed \
            --save_path ${save_dir}stage1 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch1} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.002 \
            --train_dataset_name ${obj_list[cls_id]} \
            

            CUDA_VISIBLE_DEVICES=${device} python train2.py --dataset eye_pc_3d_rgb \
            --train_data_path /data1/dengzehao/new/data/Eyecandies_processed \
            --save_path ${save_dir}stage2 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch2} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.0005 \
            --train_dataset_name ${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth
            

            CUDA_VISIBLE_DEVICES=${device} python test2.py --dataset eye_pc_3d_rgb \
            --data_path /data1/dengzehao/new/data/Eyecandies_processed \
            --save_path ./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth \
            --stage2_checkpoint_path ${save_dir}stage2/epoch_${epoch2}.pth \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --train_class ${obj_list[cls_id]}

        wait
        done
    done
done



obj_list=("ashtray0" "bag0" "bottle0")
cls_ids=(0 1 2)
for cls_id in "${!cls_ids[@]}";do
    LOG=${save_dir}"res.log"
    echo ${LOG}
    echo ${cls_id} 
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_ShapeNet
            save_dir=./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]}/weights/

            CUDA_VISIBLE_DEVICES=${device} python train.py --dataset anomalyshapenet \
            --train_data_path /data1/dengzehao/new/data/Anomaly-ShapeNet \
            --save_path ${save_dir}stage1 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch1} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.002 \
            --train_dataset_name ${obj_list[cls_id]} \
            

            CUDA_VISIBLE_DEVICES=${device} python train2.py --dataset anomalyshapenet \
            --train_data_path /data1/dengzehao/new/data/Anomaly-ShapeNet \
            --save_path ${save_dir}stage2 \
            --features_list 24 --image_size 336  --print_freq 1 \
            --batch_size 4 \
            --epoch ${epoch2} --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --learning_rate 0.0001 \
            --train_dataset_name ${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth
            

            CUDA_VISIBLE_DEVICES=${device} python test_only_point2.py --dataset anomalyshapenet \
            --data_path /data1/dengzehao/new/data/Anomaly-ShapeNet \
            --save_path ./my_exp/${times}/exps_${base_dir}_336_4/${obj_list[cls_id]} \
            --stage1_checkpoint_path ${save_dir}stage1/epoch_${epoch1}.pth \
            --stage2_checkpoint_path ${save_dir}stage2/epoch_${epoch2}.pth \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
            --train_class ${obj_list[cls_id]}
        wait
        done
    done
done

