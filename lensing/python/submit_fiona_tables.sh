#!/bin/bash
#SBATCH -n 48
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p sapphire,dvorkin
#SBATCH --mem=128000
#SBATCH -o fiona_tables_%j.out
#SBATCH -e fiona_tables_%j.err
# Builds ../assets/fiona/ (see build_fiona_tables.py). Usage, from lensing/python/: sbatch submit_fiona_tables.sh [args]
cd "$SLURM_SUBMIT_DIR"
/n/home02/nephremidze/.conda/envs/glow311/bin/python -u build_fiona_tables.py --workers "$SLURM_NTASKS" "$@"
