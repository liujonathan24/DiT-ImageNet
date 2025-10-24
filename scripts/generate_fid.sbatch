#!/bin/bash
#SBATCH --nodes=1                # Request one node
#SBATCH --ntasks=1               # Request one task (core)
#SBATCH --mem=80G                # Request 80GB of memory
#SBATCH --time=8:00:00            # Set a 4-hour time limit
#SBATCH --gres=gpu:1             # Request one GPU
#SBATCH --constraint=gpu80         # Request a node with an 80GB GPU
#SBATCH --job-name=dit-eval  # A name for your job
#SBATCH --output=slurm_%j.out    # Standard output and error log (%j is the job ID)
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job start, end, or failure
#SBATCH --mail-user=jl0796@princeton.edu # Replace with your email address

# --- Your Commands Start Here ---

# 1. Print job information (optional, but helpful for debugging)
echo "Starting job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Job was submitted from: $SLURM_SUBMIT_DIR"
echo "----------------------------------------"

# 2. Load the necessary modules
module load anaconda3/2024.6

# 3. Activate your conda environment
# Important: Use 'source' or '.' for 'conda activate' in batch scripts
source activate /scratch/network/jl0796/dit-env

# 4. Run your Python training script
echo "Running Python script..."
python -m diffusion.fid_sample -n 10000

echo "----------------------------------------"
echo "Job finished with exit code $?"
