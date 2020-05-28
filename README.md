# FashionSearch
Deep Learning Model for fashion category and attribute recommendation.

Download models here: https://drive.google.com/drive/folders/1gNX_sO2kaksyZvnmwA-AkhLiyJSACJ6I
Method1: whole.pky 
Method2: resnetFixedFinal.pkl
Extra (Trained on bad data): multiresnet2Epoch.pkl

Method 1:
-src/base_networks.py
-src/dataset.py
-src/lm_networks.py
-src/networks.py
-src/train.py
-src/conf

Method 2 (bulk of original code):
-src/multitask_model.py (final train/model file)
-src/our_train.py (prior train file. Depracated but submitted just in case)
-src/predict.py
