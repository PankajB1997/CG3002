# import standard python libraries
import logging
import os, pickle, json, h5py, operator
import numpy as np
import pandas as pd
from collections import Counter

# import libraries for ML
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, Normalizer
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier

# Fix seed value for reproducibility
np.random.seed(1234)

N = 32
OVERLAP = 0.75
MDL = "_segment-" + str(N) + "_overlap-newf-" + str(OVERLAP * 100)

# initialise logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CG3002_FILEPATH = os.path.join('/', 'CG3002')

# set constant flag for which classifier to use
'''
0: OneVsRestClassifier(estimator = MLPClassifier()),
1: OneVsRestClassifier(estimator = SVC(kernel="linear", C=0.025)),
2: RandomForestClassifier(max_depth=5, n_estimators=200, max_features=1),
3: MLPClassifier(alpha=1),
4: SVC(kernel="linear", C=0.025),
'''

# set probability threshold for multibucketing
# PROB_THRESHOLD = 0.20

MODEL_UNIQUE_IDS = {
    0: 'OneVsRestClassifierMLP' + MDL,
    1: 'OneVsRestClassifierSVC' + MDL,
    2: 'RandomForestClassifier200' + MDL,
    3: 'MLPClassifier' + MDL,
    4: 'LinearSVC' + MDL,
}

CONFIDENCE_THRESHOLD = 0.75

ENC_LIST = [
    ('sidestep', 0),
    ('number7', 1),
    ('chicken', 2),
    ('wipers', 3),
    ('turnclap', 4),
    ('numbersix', 5),
    ('salute', 6),
    ('mermaid', 7),
    ('swing', 8),
    ('cowboy', 9),
    ('logout', 10)
    # ('IDLE', 11),
]

ENC_DICT = {
    0: 'sidestep',
    1: 'number7',
    2: 'chicken',
    3: 'wipers',
    4: 'turnclap',
    5: 'numbersix',
    6: 'salute',
    7: 'mermaid',
    8: 'swing',
    9: 'cowboy',
    10: 'logout'
    # 11: 'IDLE',
}

CLASSLIST = [ pair[0] for pair in ENC_LIST ]

# Obtain best class from a given list of class probabilities for every prediction
def onehot2str(onehot):
       enc_dict = dict([(i[1],i[0]) for i in ENC_LIST])
       idx_list = np.argmax(onehot, axis=1).tolist()
       result_str = []
       for i in idx_list:
               result_str.append(enc_dict[i])
       return np.asarray(result_str)

# Convert a class to its corresponding one hot vector
def str2onehot(Y):
   enc_dict = dict(ENC_LIST)
   new_Y = []
   for y in Y:
       vec = np.zeros((1,len(ENC_LIST)),dtype='float64')
       vec[ 0, enc_dict[y] ] = 1.
       new_Y.append(vec)
   del Y
   new_Y = np.vstack(new_Y)
   return new_Y

# Computes precision, recall and F1 scores for every class
def precision_recall_f1(Y_pred, Y_test, classlist):
    precision = precision_score(Y_test, Y_pred, average=None, labels=classlist)
    recall = recall_score(Y_test, Y_pred, average=None, labels=classlist)
    f1 = f1_score(Y_test, Y_pred, average=None, labels=classlist)
    metrics = {}
    for i in range(0, len(classlist)):
        metrics[classlist[i]] = { 'precision': precision[i], 'recall': recall[i], 'f1': f1[i] }
    return metrics

# Computes micro, macro and weighted values for precision, recall and f1 scores
# micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.
# macro: Calculate metrics for each label, and find their unweighted mean.
# weighted: Calculate metrics for each label, and find their average, weighted by label imbalance.
def micro_macro_weighted(Y_pred, Y_true):
    results = {}
    results['micro_precision'] = precision_score(Y_true, Y_pred, average='micro')
    results['macro_precision'] = precision_score(Y_true, Y_pred, average='macro')
    results['weighted_precision'] = precision_score(Y_true, Y_pred, average='weighted')
    results['micro_recall'] = recall_score(Y_true, Y_pred, average='micro')
    results['macro_recall'] = recall_score(Y_true, Y_pred, average='macro')
    results['weighted_recall'] = recall_score(Y_true, Y_pred, average='weighted')
    results['micro_f1'] = f1_score(Y_true, Y_pred, average='micro')
    results['macro_f1'] = f1_score(Y_true, Y_pred, average='macro')
    results['weighted_f1'] = f1_score(Y_true, Y_pred, average='weighted')
    return results

# Calculate and display various accuracy, precision, recall and f1 scores
def calculatePerformanceMetrics(Y_pred, Y_true, dataset_type):
    assert len(Y_pred) == len(Y_true)

    num_incorrect = len(Y_true) - accuracy_score(Y_true, Y_pred, normalize=False)
    logger.info(len(Y_true))
    logger.info(accuracy_score(Y_true, Y_pred, normalize=False))
    accuracy = accuracy_score(Y_true, Y_pred)
    metrics = precision_recall_f1(Y_pred, Y_true, CLASSLIST)
    # micro_macro_weighted_scores = micro_macro_weighted(Y_pred, Y_true)
    cf_matrix = confusion_matrix(Y_true, Y_pred, labels=CLASSLIST)

    logger.info("Results for " + dataset_type + " set...")
    logger.info("Number of cases that were incorrect: " + str(num_incorrect))
    logger.info("Accuracy: " + str(accuracy))

    for i in range(0, len(CLASSLIST)):
        # logger.info("Precision " + CLASSLIST[i] + ": " + str(metrics[CLASSLIST[i]]['precision']))
        logger.info("Recall " + CLASSLIST[i] + ": " + str(metrics[CLASSLIST[i]]['recall']))
        # logger.info("F1 " + CLASSLIST[i] + ": " + str(metrics[CLASSLIST[i]]['f1']))

    # logger.info("Micro precision: " + str(micro_macro_weighted_scores['micro_precision']))
    # logger.info("Micro recall: " + str(micro_macro_weighted_scores['micro_recall']))
    # logger.info("Micro f1: " + str(micro_macro_weighted_scores['micro_f1']))
    #
    # logger.info("Macro precision: " + str(micro_macro_weighted_scores['macro_precision']))
    # logger.info("Macro recall: " + str(micro_macro_weighted_scores['macro_recall']))
    # logger.info("Macro f1: " + str(micro_macro_weighted_scores['macro_f1']))
    #
    # logger.info("Weighted precision: " + str(micro_macro_weighted_scores['weighted_precision']))
    # logger.info("Weighted recall: " + str(micro_macro_weighted_scores['weighted_recall']))
    # logger.info("Weighted f1: " + str(micro_macro_weighted_scores['weighted_f1']))

    logger.info("Confusion Matrix below " + str(CLASSLIST) + " : ")
    logger.info("\n" + str(cf_matrix))

# Obtain a list of class probability values for every prediction
def recordClassProbabilites(pred):
    class_probabilities = []
    for i in range(0, len(pred)):
        prob_per_sentence = {}
        for j in range(0, len(pred[i])):
            prob_per_sentence[ENC_DICT[j]] = pred[i][j]
        class_probabilities.append(prob_per_sentence)
    return class_probabilities

# Record model confidence on every prediction
def calculatePredictionConfidence(pred):
    CONFIDENCE_THRESHOLD = 0.65
    confidence_list = []
    for probs in pred:
        if max(probs) > CONFIDENCE_THRESHOLD:
            confidence_list.append("YES")
        else:
            confidence_list.append("NO")
    return confidence_list

# # Prepare a detailed log of all incorrect cases
# def logIncorrectCases(..., appendFileNameString):
#     ...

# Write a list of label and sentence pairs to excel file
def writeDatasetToExcel(X, y, filepath):
    df = pd.DataFrame(
        {
            'Label': y,
            'Text': X
        })
    writer = pd.ExcelWriter(filepath)
    df.to_excel(writer, "Sheet1", index=False)
    writer.save()

# Obtain a list of all classes for each prediction for which probability is greater than a threshold
# def prob2str_multibucket(probs,sens):
#     enc_dict = dict([(i[1],i[0]) for i in ENC_LIST])
#     cats = []
#     final_sens = []
#     for (prob,sen) in zip(probs,sens):
#         classes = ""
#         for idx,pro in enumerate(prob):
#             if pro >= PROB_THRESHOLD:
#                 classes += enc_dict[idx] + ", "
#         cats.append(classes[:-2])
#         final_sens.append(sen)
#     return np.asarray(cats), final_sens

# Initialise neural network model using classifier
def initialiseModel(model_index):
    classifiers = [
        OneVsRestClassifier(estimator = MLPClassifier()),
        OneVsRestClassifier(estimator = SVC(kernel="linear", C=0.025)),
        RandomForestClassifier(max_depth=5, n_estimators=200, max_features=1),
        MLPClassifier(alpha=1),
        SVC(kernel="linear", C=0.025),
    ]
    return classifiers[model_index]

# Train the model, cross-validate and save on file
def fitModel(X, Y):
    models = []
    scores = []

    for i in range(0, 5):
        model = initialiseModel(i)
        accuracy_scores = cross_val_score(model, X, Y, cv=5, scoring="accuracy", n_jobs=-1)
        scores.append(accuracy_scores.mean())
        logger.info("Cross validation score for model " + str(MODEL_UNIQUE_IDS[i]) + ": " + str(accuracy_scores.mean()))
        model.fit(X, Y)
        filepath = os.path.join("classifier_models", "model_" + MODEL_UNIQUE_IDS[i] + ".pkl")
        pickle.dump(model, open(filepath, 'wb'))
        models.append(model)

    max_index = 0
    max_accuracy_score = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > max_accuracy_score:
            max_accuracy_score = scores[i]
            max_index = i
    logger.info("Best model is " + str(MODEL_UNIQUE_IDS[max_index]) + " with accuracy of " + str(max_accuracy_score))

    return models[max_index]

def filterDataset(X, Y, X_test, Y_test):
    classes_removed = [
    # No classes need to be removed from self-collected dataset unless experimenting
        'IDLE',
    ]

    del_idx = [ idx for idx, val in enumerate(Y) if val in classes_removed ]
    X = np.delete(X, del_idx, axis=0)
    Y = np.delete(Y, del_idx)

    del_idx = [ idx for idx, val in enumerate(Y_test) if val in classes_removed ]
    X_test = np.delete(X_test, del_idx, axis=0)
    Y_test = np.delete(Y_test, del_idx)

    return X, Y, X_test, Y_test

# Print model confidence for every correct prediction
def printConfidenceForCorrectPredictions(pred_prob, pred_lbl, true_lbl):
    pred_prob, pred_lbl, true_lbl = zip(*sorted(zip(pred_prob, pred_lbl, true_lbl), key = lambda x: (x[2], x[0], x[1])))
    with open(os.path.join('logs', 'correct_predictions_log.txt'), 'w') as logfile:
        confident_correct_pred_count = 0
        correct_pred_count = 0
        for i in range(len(pred_prob)):
            if pred_lbl[i] == true_lbl[i]:
                logfile.write("The actual move " + str(true_lbl[i]) + " was predicted as " + str(pred_lbl[i]) + " with a prediction confidence of " + str(pred_prob[i]) + ".\n")
                if pred_prob[i] > CONFIDENCE_THRESHOLD:
                    confident_correct_pred_count += 1
                correct_pred_count += 1
        logfile.write("\nTotal correct predictions made with " + str(CONFIDENCE_THRESHOLD * 100) + "% confidence or more: " + str(confident_correct_pred_count))
        logfile.write("\nTotal correct predictions made overall: " + str(correct_pred_count))
        logfile.write("\nTotal number of test cases: " + str(len(pred_prob)))

# Print model confidence for every wrong prediction
def printConfidenceForIncorrectPredictions(pred_prob, pred_lbl, true_lbl):
    pred_prob, pred_lbl, true_lbl = zip(*sorted(zip(pred_prob, pred_lbl, true_lbl), key = lambda x: (x[2], x[0], x[1])))
    with open(os.path.join('logs', 'incorrect_predictions_log.txt'), 'w') as logfile:
        confident_mistake_count = 0
        incorrect_pred_count = 0
        for i in range(len(pred_prob)):
            if pred_lbl[i] != true_lbl[i]:
                logfile.write("The actual move " + str(true_lbl[i]) + " was predicted as " + str(pred_lbl[i]) + " with a prediction confidence of " + str(pred_prob[i]) + ".\n")
                if pred_prob[i] > CONFIDENCE_THRESHOLD:
                    confident_mistake_count += 1
                incorrect_pred_count += 1
        logfile.write("\nTotal mistakes made with " + str(CONFIDENCE_THRESHOLD * 100) + "% confidence or more: " + str(confident_mistake_count))
        logfile.write("\nTotal mistakes made overall: " + str(incorrect_pred_count))
        logfile.write("\nTotal number of test cases: " + str(len(pred_prob)))


TRAIN_DATASET_PATH = os.path.join("dataset", "train.pkl")
TEST_DATASET_PATH = os.path.join("dataset", "test.pkl")

if __name__ == "__main__":

    # Normalizer() works best with GammaSVC
    # QuantileTransformer(output_distribution='uniform') works best with LinearSVC
    # scaler = QuantileTransformer(output_distribution='uniform')
    scaler = StandardScaler()
    # scaler = MinMaxScaler((-1,1))

    # Use the dataset prepared from self-collected dataset's raw data values
    X, Y = pickle.load(open(TRAIN_DATASET_PATH, 'rb'))
    X_test, Y_test = pickle.load(open(TEST_DATASET_PATH, 'rb'))
    X, Y, X_test, Y_test = filterDataset(X, Y, X_test, Y_test)

    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    pickle.dump(scaler, open(os.path.join('scaler', 'standard_scaler' + MDL + '.pkl'), 'wb'))

    logger.info(str(Counter(Y)))
    # logger.info(str(Counter(Y_val)))
    logger.info(str(Counter(Y_test)))

    logger.info("Vectorizing...")

    # # Do some preprocess vectorizing for training/validation/testing sets respectively, as needed
    # vectorizer(...)
    # vectorizer(...)

    logger.info("Fitting...")

    # X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42, shuffle=True, stratify=Y_test)

    model = fitModel(X, Y)

    logger.info("Predicting...")
    train_pred = model.predict(X)
    # val_pred = model.predict(X_val)
    import time
    start = time.time()
    test_pred = model.predict(X_test)
    end = time.time()

    logger.info("Prediction time: " + str(end-start))

    logger.info("Predictions done! Compiling results...")

    # # Convert model output of class probabilities to corresponding best predictions
    # Y_train_pred = onehot2str(train_pred)
    # # Y_val_pred = onehot2str(val_pred)
    # Y_test_pred = onehot2str(test_pred)

    # Calculate accuracy, precision, recall and f1 scores
    calculatePerformanceMetrics(train_pred, Y, "training")
    # calculatePerformanceMetrics(val_pred, Y_val, "validation")
    calculatePerformanceMetrics(test_pred, Y_test, "testing")

    # pred_prob = train_pred.tolist() + test_pred.tolist()
    # pred_lbl = Y_train_pred.tolist() + Y_test_pred.tolist()
    # true_lbl = Y.tolist() + Y_test.tolist()
    # for i in range(len(pred_prob)):
    #     pred_prob[i] = max(pred_prob[i])
    # # Print model confidence for every wrong prediction
    # printConfidenceForIncorrectPredictions(pred_prob, pred_lbl, true_lbl)
    # # Print model confidence for every correct prediction
    # printConfidenceForCorrectPredictions(pred_prob, pred_lbl, true_lbl)
