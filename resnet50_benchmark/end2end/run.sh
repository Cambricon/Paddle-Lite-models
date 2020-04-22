#!/bin/bash     
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:/home/zhaocai/Paddle-Lite/build.lite.mlu/third_party/install/mklml/lib/:${LD_LIBRARY_PATH}
$model_path=$1
$filelist_path=$2
for((core_num=16;core_num>=2;core_num/=2))
do
        for((batchsize=16;batchsize>=1;batchsize/=2))
        do
           for((i=1;i<=16;i++))
           do
            ./right_$i $batchsize $core_num $i 0 $model_path $filelist_path
            ./right_$i $batchsize $core_num $i 1 $model_path $filelist_path
            ./right_$i $batchsize $core_num $i 2 $model_path $filelist_path
           done       
 

        done

done

