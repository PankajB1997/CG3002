# import standard python libraries
import logging
import os, pickle, json, h5py, operator
import numpy as np
import pandas as pd
from collections import Counter

# import libraries for ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, Normalizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from keras.models import load_model, Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint

# initialise logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CG3002_FILEPATH = "\\Users\\pankaj\\Documents\\CG3002"
# "\\Users\\pankaj\\Documents\\CG3002"
SAVE_FILEPATH = "dummy_dataset\\RawData_ByMove\\"

PROB_THRESHOLD = 0.20

ENC_LIST = [
    ('WALKING', 0),
    ('WALKING_UPSTAIRS', 1),
    ('WALKING_DOWNSTAIRS', 2),
    # ('SITTING', 3),
    # ('STANDING', 4),
    # ('LAYING', 5),
    # ('STAND_TO_SIT', 6),
    # ('SIT_TO_STAND', 7),
    # ('SIT_TO_LIE', 8),
    # ('LIE_TO_SIT', 9),
    # ('STAND_TO_LIE', 10),
    # ('LIE_TO_STAND', 11)
]

CLASSLIST = [ pair[0] for pair in ENC_LIST ]

ENC_DICT = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING',
    6: 'STAND_TO_SIT',
    7: 'SIT_TO_STAND',
    8: 'SIT_TO_LIE',
    9: 'LIE_TO_SIT',
    10: 'STAND_TO_LIE',
    11: 'LIE_TO_STAND'
}

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
    logger.info(str(cf_matrix))

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
def prob2str_multibucket(probs,sens):
    enc_dict = dict([(i[1],i[0]) for i in ENC_LIST])
    cats = []
    final_sens = []
    for (prob,sen) in zip(probs,sens):
        classes = ""
        for idx,pro in enumerate(prob):
            if pro >= PROB_THRESHOLD:
                classes += enc_dict[idx] + ", "
        cats.append(classes[:-2])
        final_sens.append(sen)
    return np.asarray(cats), final_sens

# Initialise neural network model using Keras
def initialiseModel(X_train):
    main_input = Input(shape=(X_train[0].size,))
    x = Dense(512)(main_input)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    output = Dense(3, activation = 'softmax')(x)
    model = Model(inputs = main_input, outputs = output)
    # from keras import optimizers
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# Train the model, monitor on validation loss and save the best model out of given epochs
def fitModel(X_train, Y_train, X_val, Y_val):
    # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, shuffle=True)
    model = initialiseModel(X_train)
    filepath = os.path.join("nn_models", "nn_model.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    sample_weight = compute_sample_weight('balanced', Y_train)
    model.fit(X_train, str2onehot(Y_train), epochs=100, validation_data=(X_val, str2onehot(Y_val)), batch_size=50, callbacks=callbacks_list, sample_weight=sample_weight)
    return model

def loadDataset(X_PATH, Y_PATH):
    X = []
    Y = []
    with open(X_PATH) as x_file:
        for input in x_file:
            X.append(list(map(float, input[:-1].split(" "))))
    with open(Y_PATH) as y_file:
        for input in y_file:
            Y.append(ENC_DICT[int(input) - 1])
    classes_removed = [
        'STAND_TO_SIT',
        'SIT_TO_STAND',
        'SIT_TO_LIE',
        'LIE_TO_SIT',
        'STAND_TO_LIE',
        'LIE_TO_STAND'
    ]

    del_idx = [ idx for idx, val in enumerate(Y) if val in classes_removed ]
    X = np.delete(X, del_idx, axis=0)
    Y = np.delete(Y, del_idx)
    return X, Y

def filterDataset(X, Y, X_test, Y_test):
    classes_removed = [
        # 'WALKING',
        # 'WALKING_UPSTAIRS',
        # 'WALKING_DOWNSTAIRS',
        'STANDING',
        'LAYING',
        'SITTING',
        'STAND_TO_SIT',
        'SIT_TO_STAND',
        'SIT_TO_LIE',
        'LIE_TO_SIT',
        'STAND_TO_LIE',
        'LIE_TO_STAND'
    ]

    del_idx = [ idx for idx, val in enumerate(Y) if val in classes_removed ]
    X = np.delete(X, del_idx, axis=0)
    Y = np.delete(Y, del_idx)

    del_idx = [ idx for idx, val in enumerate(Y_test) if val in classes_removed ]
    X_test = np.delete(X_test, del_idx, axis=0)
    Y_test = np.delete(Y_test, del_idx)

    return X, Y, X_test, Y_test

X_TRAIN_TXT_PATH = os.path.join(CG3002_FILEPATH, "Raspberry_Pi\\dummy_dataset\\Train\\X_train.txt")
Y_TRAIN_TXT_PATH = os.path.join(CG3002_FILEPATH, "Raspberry_Pi\\dummy_dataset\\Train\\y_train.txt")
X_TEST_TXT_PATH = os.path.join(CG3002_FILEPATH, "Raspberry_Pi\\dummy_dataset\\Test\\X_test.txt")
Y_TEST_TXT_PATH = os.path.join(CG3002_FILEPATH, "Raspberry_Pi\\dummy_dataset\\Test\\y_test.txt")

if __name__ == "__main__":

    # scaler = QuantileTransformer(output_distribution='uniform')
    scaler = StandardScaler()

    # X, Y = loadDataset(X_TRAIN_TXT_PATH, Y_TRAIN_TXT_PATH)
    # X_test, Y_test = loadDataset(X_TEST_TXT_PATH, Y_TEST_TXT_PATH)
    X, Y = pickle.load(open(SAVE_FILEPATH + 'train.pkl', 'rb'))
    X_test, Y_test = pickle.load(open(SAVE_FILEPATH + 'test.pkl', 'rb'))
    X, Y, X_test, Y_test = filterDataset(X, Y, X_test, Y_test)

    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    logger.info("Vectorizing...")

    # # Do some preprocess vectorizing for training/validation/testing sets respectively, as needed
    # vectorizer(...)
    # vectorizer(...)

    logger.info("Fitting...")

    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42, shuffle=True, stratify=Y_test)

    logger.info(str(Counter(Y)))
    logger.info(str(Counter(Y_val)))
    logger.info(str(Counter(Y_test)))

    model = fitModel(X, Y, X_val, Y_val)

    logger.info("Predicting...")
    train_pred = model.predict(X)
    val_pred = model.predict(X_val)
    import time
    start = time.time()
    test_pred = model.predict(X_test)
    end = time.time()
    logger.info("Prediction time: " + str(end-start))

    logger.info("Predictions done! Compiling results...")

    # Convert model output of class probabilities to corresponding best predictions
    Y_train_pred = onehot2str(train_pred)
    Y_val_pred = onehot2str(val_pred)
    Y_test_pred = onehot2str(test_pred)

    # Calculate accuracy, precision, recall and f1 scores
    calculatePerformanceMetrics(Y_train_pred, Y, "training")
    calculatePerformanceMetrics(Y_val_pred, Y_val, "validation")
    calculatePerformanceMetrics(Y_test_pred, Y_test, "testing")

    # Record model confidence on every prediction
    train_confidence_list = calculatePredictionConfidence(train_pred)
    val_confidence_list = calculatePredictionConfidence(val_pred)
    test_confidence_list = calculatePredictionConfidence(test_pred)

    # Record class probabilities for every prediction
    train_dict_list = recordClassProbabilites(train_pred)
    val_dict_list = recordClassProbabilites(val_pred)
    test_dict_list = recordClassProbabilites(test_pred)

    # # Prepare a detailed log of all incorrect cases on every prediction as text file
    # logIncorrectCases(..., 'training-crossval')
    # logIncorrectCases(..., 'testing')
