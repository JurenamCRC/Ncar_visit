#!/bin/bash -l

## Job Name
#PBS -N py_job
### Charging account
#PBS -A NCGD0011
### Request one chunk of resources with 1 CPU and 10 GB of memory
#PBS -l select=1:ncpus=1:mem=10GB
### Allow job to run up to 30 minutes
#PBS -l walltime=23:00:00
### Route the job to the casper queue
#PBS -q casper
### Join output and error streams into single file


# Run 
module load conda/latest
conda activate npl
python /glade/u/home/pmora/mean_HOURLY.py
