export DIR="$(dirname "$(pwd)")"
conda env update --file ${DIR}'/uncertainty_guided_env.yml'
source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}

export train_path=${DIR}'/JTVAE/data/zinc/train.txt'
export vocab_path=${DIR}'/JTVAE/data/zinc/new_vocab.txt'
export prop_path=${DIR}'/JTVAE/data/zinc/train.logP-SA'
export save_path=${DIR}'/JTVAE/checkpoints/jtvae_drop_MLP0.2_GRU0.2_Prop0.2_zdim56_hidden450_prop-logP'

export bs=32
export dropout_rate_MLP=0.2
export dropout_rate_GRU=0.2

export hidden_size=450
export latent_size=56

export property="penalized_logP"
export drop_prop_NN=0.2

python ../JTVAE/fast_molvae/jtnnvae-prop_pretrain.py \
            --train_path ${train_path} \
            --vocab_path  ${vocab_path} \
            --prop_path ${prop_path} \
            --save_path ${save_path} \
            --batch_size ${bs} \
            --hidden_size ${hidden_size} \
            --latent_size ${latent_size} \
            --dropout_rate_MLP ${dropout_rate_MLP} \
            --dropout_rate_GRU ${dropout_rate_GRU} \
            --drop_prop_NN ${drop_prop_NN} \
            --property ${property}