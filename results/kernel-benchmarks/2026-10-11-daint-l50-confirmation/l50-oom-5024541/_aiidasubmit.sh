#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="aiida-13212"
#SBATCH --get-user-env
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=2
#SBATCH --time=14:00:00
#SBATCH --mem=262144

#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=lp83


#SBATCH --uenv=julia/26.3:v1
#SBATCH --view=juliaup
#SBATCH --repo=/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/uenv-repo
export JULIA_DEPOT_PATH=/ritom/scratch/cscs/cpi/goncalo/updatespin_l23_validation_20260917/depot:/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/depot
export ITENSOR_SPIN_CASE_ROOT=/ritom/scratch/cscs/cpi/goncalo/aiida_itensor_spin_0.4.3_bondcap_20261005/engine
export ITENSOR_SPIN_JULIA=/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/julia-1.12.6/bin/julia
export GKSwstype=100
ulimit -s unlimited
(cd /ritom/scratch/cscs/cpi/goncalo/aiida_itensor_spin_0.4.3_bondcap_20261005 && printf '%s  SHA256SUMS\n' 739f114a37287091acfcc68c7b97c5b35aa84741075fa9579a2b1eee84a41191 | sha256sum -c - && sha256sum -c SHA256SUMS) || exit 90

'/ritom/scratch/cscs/cpi/goncalo/aiida_itensor_spin_0.4.3_bondcap_20261005/engine/itensor-spin-segment' 'segment.json'  > 'scheduler-stdout.txt' 2> 'scheduler-stderr.txt'
