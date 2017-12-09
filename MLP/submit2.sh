#!/bin/bash
 
#BSUB -n 2
#BSUB -R "span[ptile=2]"
#BSUB -R gpu
#BSUB -q "windfall"
#BSUB -o MLP_GPU_2_Batch_500.out
#BSUB -e MLP_GPU_2_Batch_500.err
#BSUB -J MLP_GPU_2_Batch_500
 
#---------------------------------------------------------------------
module load singularity/2.3.1
cd /home/u15/zhengzhongliang/TensorflowGPU/MLP_GPU_2
singularity run --nv /home/u15/zhengzhongliang/TensorflowGPU/tf_gpu-1.2.0-cp35-cuda8-cudnn51.img MLP_GPU_2_Batch_500.py
