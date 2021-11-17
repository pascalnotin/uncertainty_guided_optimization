export DIR="$(dirname "$(pwd)")"
conda env update --file ${DIR}'/uncertainty_guided_env.yml'
source activate uncertainty_guided_env
export PYTHONPATH=${PYTHONPATH}:${DIR}

export processed_data_path=${DIR}'/JTVAE/data/zinc_processed'
export save_path=${DIR}'/JTVAE/checkpoints/jtvae_drop_MLP0.2_GRU0.2_zdim56_hidden450/'
export vocab_path=${DIR}'/JTVAE/data/zinc/new_vocab.txt'

export bs=32
export dropout_rate_MLP=0.2
export dropout_rate_GRU=0.2
export hidden_size=450
export latent_size=56

python ../JTVAE/fast_molvae/jtnnvae_train.py \
            --train_path ${processed_data_path} \
            --vocab_path  ${vocab_path} \
            --save_path ${save_path} \
            --batch_size ${bs} \
            --hidden_size ${hidden_size} \
            --latent_size ${latent_size} \
            --dropout_rate_MLP ${dropout_rate_MLP} \
            --dropout_rate_GRU ${dropout_rate_GRU}