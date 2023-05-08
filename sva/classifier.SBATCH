#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --job-name=classifier
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yq810@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --mem=48GB

module purge

singularity exec --nv \
	    --overlay /scratch/yq810/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/yq810/interpret_bert/sva; python classifier.py --input_folder ./output_fold --layer 0"
