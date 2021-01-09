# Articulatory-GAN

![articulatory_gif]()

The purpose of this project is to examine whether the deep-learning based generative adversarial networks (GAN) can effectively learn the representation of the human speech articulation. Especially, if the mid-sagittal articulatory configurations for vowels are provided, can the generator in GAN reasonably learn to represent vowel articulation? Whether it is reasonable or not, will be tested both qualitatively (eg. reconstructed articulatory configurations) and quantitively (eg. using a vowel classifier model).

- The aim of this project: "Can the generator learn the vowel articulatory representations?"
- Specific goals:
    - [ ] (1) Train GAN and cGAN models
    - [ ] (2) Test the trained models with interactive visualizations

## Procedure
- (1) Prepare data
- (2) Train the model
- (3) Test the model

## Requirements
```bash
# Data
Preprocessed articualtory data: See https://github.com/jaekookang/Articulatory-Data-Extractor

# Modules
tensorflow
...
```

## Reference
- Haskins IEEE EMA dataset
- Python-EMA-Viewer
- Articulatory-Data-Extractor

