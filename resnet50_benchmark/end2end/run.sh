#!/bin/bash
for((core_num=16;core_num>=2;core_num/=2))
do
        for((batchsize=16;batchsize>=1;batchsize/=2))
        do
           for((i=1;i<=16;i++))
           do
            ./right_$i $batchsize $core_num $i 0 $1 $2
            ./right_$i $batchsize $core_num $i 1 $1 $2
            ./right_$i $batchsize $core_num $i 2 $1 $2
           done       
 

        done

done

