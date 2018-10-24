# import python libraries
# mpl usage here is for some funky workaround for using matplotlib with mac osx
# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
import logging
import os, pickle, json, h5py, operator
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates

# initialise logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATASET_PATH = "dataset/"

# moves = ['idle', 'logout', 'number_six']
#
# data_by_move = {}

# i = 1
#
# for move in moves:
#     move_data_current_dancer = DATASET_PATH + move + '.pkl'
#     if os.path.exists(move_data_current_dancer):
#         X, y = pickle.load(open(move_data_current_dancer, 'rb'))
        # pca = sklearnPCA(n_components=2) #2-dimensional PCA
        # transformed = pd.DataFrame(pca.fit_transform(X))
#         plt.scatter(transformed[y==i][0], transformed[y==i][1], label=move)
#     i += 1

logger.info(pickle.load(open(DATASET_PATH + 'number_six' + '.pkl', 'rb')))

X, y = pickle.load(open(DATASET_PATH + 'number_six' + '.pkl', 'rb'))

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X, y))

plt.scatter(transformed[0], transformed[1], label='number_six', c='red')

X, y = pickle.load(open(DATASET_PATH + 'idle' + '.pkl', 'rb'))

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X, y))

plt.scatter(transformed[0], transformed[1], label='idle', c='blue')

X, y = pickle.load(open(DATASET_PATH + 'logout' + '.pkl', 'rb'))
#
# logger.info(X)
# logger.info(y)
#
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X, y))
#
# logger.info("transformed")
# logger.info(transformed)
#
plt.scatter(transformed[0], transformed[1], label='logout', c='green')

# plot
plt.legend()
plt.show()
