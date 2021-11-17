export DIR="$(dirname "$(pwd)")"
conda env update --file ${DIR}'/uncertainty_guided_env.yml'
source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}

export experiment_name="20210101"
export model_type='JTVAE'
export model_checkpoint=${DIR}'/JTVAE/checkpoints/jtvae_drop_MLP0.2_GRU0.2_Prop0.2_zdim56_hidden450_prop-logP/model.final'
export vocab_path=${DIR}'/JTVAE/data/zinc/new_vocab.txt'
export mode_generation_starting_point="random"  #[random|low_property_objects]

export batch_size=100
export num_starting_points=100
export optimization_method="gradient_ascent"

export GA_number_optimization_steps=100
export GA_alpha=200.0
export GA_uncertainty_threshold="max" #[No_constraint|max|P99|P95|P90]

export decoder_uncertainty_method="MI_Importance_sampling" #[MI_Importance_sampling|NLL_prior]
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
            --GA_number_optimization_steps ${GA_number_optimization_steps} \
            --GA_alpha ${GA_alpha} \
            --GA_uncertainty_threshold ${GA_uncertainty_threshold} \
            --decoder_uncertainty_method ${decoder_uncertainty_method} \
            --decoder_num_sampled_models ${decoder_num_sampled_models} \
            --decoder_num_sampled_outcomes ${decoder_num_sampled_outcomes} \
            --seed ${seed}