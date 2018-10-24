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

# initialise logger test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATASET_PATH = "dataset/"

X1, y1 = pickle.load(open(DATASET_PATH + 'idle' + '.pkl', 'rb'))
X2, y2 = pickle.load(open(DATASET_PATH + 'logout' + '.pkl', 'rb'))
X3, y3 = pickle.load(open(DATASET_PATH + 'number_six' + '.pkl', 'rb'))

X = X1 + X2 + X3
y = y1 + y2 + y3

# logger.info(X, y)

lda = LDA(n_components=2) #2-dimensional LDA

# logger.info(lda)
logger.info("lda")

y = pd.Series(y)
logger.info(y)


logger.info(lda.fit_transform(X, y))

lda_transformed = pd.DataFrame(lda.fit_transform(X, y))

# logger.info(lda_transformed)
logger.info("lda_transformed")

plt.scatter(lda_transformed[y=='idle'][0], lda_transformed[y=='idle'][1], label='idle', c='red')
plt.scatter(lda_transformed[y=='logout'][0], lda_transformed[y=='logout'][1], label='logout', c='blue')
plt.scatter(lda_transformed[y=='number_six'][0], lda_transformed[y=='number_six'][1], label='number_six', c='green')
#
# X, y = pickle.load(open(DATASET_PATH + 'idle' + '.pkl', 'rb'))
#
# lda = LDA(n_components=2) #2-dimensional LDA
# lda_transformed = pd.DataFrame(lda.fit_transform(X, y))
#
# plt.scatter(lda_transformed[0], lda_transformed[1], label='idle', c='blue')
#
# X, y = pickle.load(open(DATASET_PATH + 'logout' + '.pkl', 'rb'))
# #
# # logger.info(X)
# # logger.info(y)
# #
# lda = LDA(n_components=2) #2-dimensional LDA
# lda_transformed = pd.DataFrame(lda.fit_transform(X, y))
# #
# # logger.info("transformed")
# # logger.info(transformed)
# #
# plt.scatter(lda_transformed[0], lda_transformed[1], label='logout', c='green')

# plot
plt.legend()
plt.show()
