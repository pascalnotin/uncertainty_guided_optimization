export DIR="$(dirname "$(pwd)")"
conda env update --file ${DIR}'/uncertainty_guided_env.yml'
source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}

export experiment_name="20210101"
export model_type='JTVAE'
export model_checkpoint=${DIR}'/JTVAE/checkpoints/jtvae_drop_MLP0.2_GRU0.2_Prop0.2_zdim56_hidden450_prop-logP/model.final'
export vocab_path=${DIR}'/JTVAE/data/zinc/new_vocab.txt'
export mode_generation_starting_point="random"  #[random|low_property_objects]

export batch_size=500
export num_starting_points=500
export optimization_method="bayesian_optimization"

export BO_number_objects_generated=500
export BO_uncertainty_mode="Uncertainty_censoring"
export BO_uncertainty_threshold="max" #[No_constraint|max|P99|P95|P90]
export BO_abs_bound=20.0
export BO_acquisition_function="qEI"
export BO_default_value_invalid=-2.0 #Low value (P5 on training data)

export decoder_uncertainty_method="NLL_prior" #[MI_Importance_sampling|NLL_prior]
export decoder_num_sampled_models=100
export decoder_num_sampled_outcomes=1

export seed=2021

python ../uncertainty_guided_optimization.py \
            --experiment_name ${experiment_name} \
            --model_type ${model_type} \
            --model_checkpoint ${model_checkpoint} \
            --vocab_path ${vocab_path} \
            --mode_generation_starting_point ${mode_generation_starting_point} \
            --num_starting_points ${num_starting_points} \
            --batch_size ${batch_size} \
            --optimization_method ${optimization_method} \
            --BO_number_objects_generated ${BO_number_objects_generated} \
            --BO_uncertainty_mode ${BO_uncertainty_mode} \
            --BO_uncertainty_threshold ${BO_uncertainty_threshold} \
            --BO_abs_bound ${BO_abs_bound} \
            --BO_acquisition_function ${BO_acquisition_function} \
            --BO_default_value_invalid ${BO_default_value_invalid} \
            --decoder_uncertainty_method ${decoder_uncertainty_method} \
            --decoder_num_sampled_models ${decoder_num_sampled_models} \
            --decoder_num_sampled_outcomes ${decoder_num_sampled_outcomes} \
            --seed ${seed}