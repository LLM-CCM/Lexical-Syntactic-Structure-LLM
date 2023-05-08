#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=fullRun
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yq810@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --mem=64GB

module purge

singularity exec --nv \
	    --overlay /scratch/yq810/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/yq810/interpret_bert/sva; python torch-test.py;  python extract_features.py --data_file fulldata.tsv --num_gpus 1 --output_folder ./output_fold"
