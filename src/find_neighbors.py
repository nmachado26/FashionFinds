from sklearn.cluster import KMeans
import numpy as np
import const
import pandas as pd
import sys


#IMPORTANT: Added this separate file for a clean separation of the nearest neighbors function.
#for readability. This is not functional due to missing imports/subcode
#Due to ease of dependency and superclass use, the functional implementation (copy) is in
#train.py
if __name__ == '__main__':
    #nearest neighbors implementation

    df = pd.read_csv(base_path + const.USE_CSV)
    train_df = df[df['evaluation_status'] == 'train']
    X = df['image_name']
    Y_list = list(df)
    Y_list.remove('image_name')
    y = df[Y_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    attributes = ['attr_%d'%i for i in range(1000)]
    attr_train = y_train[attributes]
    attr_nodes = attr_train.shape[1]
    attr_train = attr_train.values.tolist()

    #hardcoded an image prediction of animal and animal print attributes turned on in one hot. Replace "indexes" with the converted predicted attribute array
    indexes = [18,19]
    test_arr = []
    for i in range(1000):
        if i in indexes:
            test_arr.append(1)
        else:
         test_arr.append(0)
    test_arr_t = torch.from_numpy(numpy.array(test_arr)).float()

    #for norm implementation, uncomment below:
    #min_dist = float('inf')
    #min_arr = None
    #min_arrs = []
    tups = dict()
    for i, attr_arr in enumerate(attr_train):
        attr_arr_t = torch.from_numpy(numpy.array(attr_arr)).float()
        dist = numpy.sum(numpy.absolute(numpy.subtract(numpy.array(test_arr), numpy.array(attr_arr))))
        tups[i] = dist
        #for norm implementation, uncomment below:
        #dist = torch.norm(test_arr_t - attr_arr_t)
        #if dist < min_dist:
        #    tups[i] = dist
        #    min_dist = dist
        #    min_arrs.append(X[i])
        #    min_arr = i

    #print topk where k = 10
    for i, w in enumerate(sorted(tups, key=tups.get)):
        if i < 10:
            print('name: ', X[w])
            print('dist: ', tups[w])
