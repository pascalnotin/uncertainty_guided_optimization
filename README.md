# Uncertainty guided optimization
Official repository for the paper [Improving black-box optimization in VAE latent space using decoder uncertainty](https://arxiv.org/abs/2107.00096)
(Pascal Notin, José Miguel Hernández-Lobato, Yarin Gal).

## Abstract
Optimization in the latent space of variational autoencoders is a promising approach to generate high-dimensional discrete objects that maximize an expensive black-box property (e.g., drug-likeness in molecular generation, function approximation with arithmetic expressions). However, existing methods lack robustness as they may decide to explore areas of the latent space for which no data was available during training and where the decoder can be unreliable, leading to the generation of unrealistic or invalid objects. We propose to leverage the epistemic uncertainty of the decoder to guide the optimization process. This is not trivial though, as a naive estimation of uncertainty in the high-dimensional and structured settings we consider would result in high estimator variance. To solve this problem, we introduce an importance sampling-based estimator that provides more robust estimates of epistemic uncertainty. Our uncertainty-guided optimization approach does not require modifications of the model architecture nor the training process. It produces samples with a better trade-off between black-box objective and validity of the generated samples, sometimes improving both simultaneously. We illustrate these advantages across several experimental settings in digit generation, arithmetic expression approximation and molecule generation for drug design.

## Junction-Tree VAE (JTVAE)
We extend the molecular optimization approach described in [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/abs/1802.04364) by Jin et al., and build on top of the corresponding codebase: https://github.com/wengong-jin/icml18-jtnn.

This repository includes the following enhancements:
- Extending the JTNNVAE class with methods to quantify decoder uncertainty in latent.
- Creating a separate subclass (JTNNVAE_prop) to facilitate the joint training of a JTVAE with an auxiliary network predicting a property of interest (eg., penalized logP).
- Including additional hyperparameters to apply dropout in the JTVAE architecture (thereby supporting weight sampling via MC dropout).
- Providing new functionalities to perform uncertainty-guided optimization in latent and assess the quality of generated molecules.
- Supporting a more comprehensive set of Bayesian Optimization functionalities via [BoTorch](https://botorch.org/).
- Migrating the original codebase to a more recent software stack (Python v3.8 and Pytorch v1.10.0).

Example scripts are provided in `scripts/` to:
1. Preprocess the data (JTVAE_data_preprocess.sh) and generate a new vocabulary on new dataset (JTVAE_data_vocab_generation.sh)
2. Train the JTVAE networks:
- JTVAE with no auxiliary property network: JTVAE_train_jtnnvae.sh
- JTVAE with auxiliary property network: 
    - JTVAE_train_jtnnvae-prop_step1_pretrain.sh to pre-train the joint architecture (with no KL term in the loss)
    - JTVAE_train_jtnnvae-prop_step2_train.sh to train the joint architecture
3. Test the quality of trained JTVAE networks (JTVAE_test_jtnnvae.sh and JTVAE_test_jtnnvae-prop.sh)
4. Perform uncertainty-guided optimization in latent:
- Gradient ascent: JTVAE_uncertainty_guided_optimization_gradient_ascent.sh
- Bayesian optimization: JTVAE_uncertainty_guided_optimization_bayesian_optimization.sh

## Environment setup
The required environment may be created via conda and the provided uncertainty_guided_env.yml file as follows:
```
  conda env create -f uncertainty_guided_env.yml
  conda activate uncertainty_guided_env
```

## Citation
If you use this code, please cite the following [paper](https://arxiv.org/abs/2107.00096) as:
```bibtex
@misc{notin2021improving,
      title={Improving black-box optimization in VAE latent space using decoder uncertainty}, 
      author={Pascal Notin and José Miguel Hernández-Lobato and Yarin Gal},
      year={2021},
      eprint={2107.00096},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```