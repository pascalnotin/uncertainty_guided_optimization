export DIR="$(dirname "$(pwd)")"
conda env update --file ${DIR}'/uncertainty_guided_env.yml'
source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}

export nsample=100
export vocab=${DIR}'/JTVAE/data/zinc/new_vocab.txt'
export model_checkpoint=${DIR}'/JTVAE/checkpoints/jtvae_drop_MLP0.2_GRU0.2_Prop0.2_zdim56_hidden450_prop-logP/model.final'
export output_file='20210101'

export hidden_size=450
export latent_size=56
export depthT=20
export depthG=3
export dropout_rate_GRU=0.2
export dropout_rate_MLP=0.2
export property="penalized_logP"
export drop_prop_NN=0.2

python ../JTVAE/fast_molvae/sample.py \
            --nsample ${nsample} \
            --vocab  ${vocab} \
            --model_checkpoint ${model_checkpoint} \
            --output_file ${output_file} \
            --hidden_size ${hidden_size} \
            --latent_size ${latent_size} \
            --depthT ${depthT} \
            --depthG ${depthG} \
            --dropout_rate_GRU ${dropout_rate_GRU} \
            --dropout_rate_MLP ${dropout_rate_MLP} \
            --property ${property} \
            --drop_prop_NN ${drop_prop_NN}
            