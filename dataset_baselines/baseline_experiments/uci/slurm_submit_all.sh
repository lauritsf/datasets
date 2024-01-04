

log_dir="/scratch/laula/uci_baselines/logs/"
slurm_log_dir="/scratch/laula/uci_baselines/slurm_logs/"

# log_dir="logs/"
# slurm_log_dir="slurm_logs/"

datasets=(
    "boston_housing"
    "concrete_strength"
    "energy_efficiency"
    "kin8nm"
    "naval_propulsion"
    "power_plant"
    # "protein_structure"
    "wine_quality_red"
    "yacht_hydrodynamics"
)
seeds=$(seq 0 9)

# Deterministic
noise_models=("homoscedastic" "heteroscedastic")
ensemble_sizes=(1 10)

deterministic_python_script="dataset_baselines/baseline_experiments/uci/deterministic.py"
for dataset in "${datasets[@]}"; do
    for seed in $seeds; do
        for noise_model in "${noise_models[@]}"; do
            for ensemble_size in "${ensemble_sizes[@]}"; do
                job_name="uci_${dataset}_seed_${seed}_noise_model_${noise_model}_ensemble_size_${ensemble_size}"
                sbatch --job-name=$job_name \
                    --output=$slurm_log_dir$job_name.log \
                    scripts/sbatch_wrapper.sh $deterministic_python_script \
                    --dataset $dataset \
                    --seed $seed \
                    --noise_model $noise_model \
                    --ensemble_size $ensemble_size \
                    --log_dir $log_dir
                # wait a moment before submitting the next job
                sleep 0.1
            done
        done
    done
done


# Variational
layer_types=("full")
variational_python_script="dataset_baselines/baseline_experiments/uci/variational.py"
for dataset in "${datasets[@]}"; do
    for seed in $seeds; do
        for layer_type in "${layer_types[@]}"; do
            job_name="uci_${dataset}_seed_${seed}_variational_${layer_type}"
            sbatch --job-name=$job_name \
                --output=$slurm_log_dir$job_name.log \
                scripts/sbatch_wrapper.sh $variational_python_script \
                --dataset $dataset \
                --seed $seed \
                --layer_type $layer_type \
                --log_dir $log_dir
                sleep 0.1
        done
    done
done
