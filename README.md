# FashionSearch
Deep Learning Model for fashion category and attribute recommendation.

Download models here: https://drive.google.com/drive/folders/1gNX_sO2kaksyZvnmwA-AkhLiyJSACJ6I

Method1 Model: whole.pky 

Method2 Model: resnetFixedFinal.pkl

Extra Model (Trained on bad data): multiresnet2Epoch.pkl

Method 1 Files:
-src/base_networks.py
-src/dataset.py
-src/lm_networks.py
-src/networks.py
-src/train.py
-src/conf

Method 2 Files (bulk of original code):
-src/multitask_model.py (final train/model file)
-src/our_train.py (prior train file. Deprecated but submitted just in case)
-src/predict_method2.py

Final Submission Update:

Finalized predict and kmeans implementations are written inside src/train.py file, after the training loop.
