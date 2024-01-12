# Fully Connected Inceptioned U-Net
Code for the UCL MSc Data Science and Machine Learning thesis.

## Repository structure
* `datasets/blastocyst.py` contains `Dataset` classes that handle the data
* `models/unet.py` contains the baseline U-Net model
* `models/incept_unet.py` contains the proposed FC Inceptioned U-Net model
* `utils/losses.py` contains classes with custom losses
* `utils/trainer.py` contains class that handles model training, evaluation, and results visualisation
* `train_*.py` contain code to train the corresponding model and perform a grid search
* `*.ipynb` contain a more interactive code to train and/or evaluate the corresponding model

## Data
Due to data sharing restrictions, we anonymised part of the data. However, if you wish to train and evaluate the model yourself, you can use the `SFUDataset` class with a publicly available dataset introduced by the Pacific Centre for Reproductive Medicine (PCRM) [1].

## References
[1] Parvaneh Saeedi, Dianna Yee, Jason Au, and Jon Havelock. Automatic identification of human blastocyst components via texture. *IEEE Transactions on Biomedical Engineering*, 64(12):2968–2978, 2017.

## Note
Due to data sharing restrictions the commit history wasn't published publicly, since it contains sensitive data.
