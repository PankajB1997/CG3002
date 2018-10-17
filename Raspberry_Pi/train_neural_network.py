# import standard python libraries
import logging, os, pickle, json, h5py, operator, time
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

CONFIDENCE_THRESHOLD = 0.95

ENC_LIST = [
    ('IDLE', 0),
    ('wipers', 1),
    ('number7', 2),
    ('chicken', 3),
    ('sidestep', 4),
    ('turnclap', 5),
    # ('numbersix', 6),
    # ('salute', 7),
    # ('mermaid', 8),
    # ('swing', 9),
    # ('cowboy', 10),
    # ('logout', 11)
]

ENC_DICT = {
    0: 'IDLE',
    1: 'wipers',
    2: 'number7',
    3: 'chicken',
    4: 'sidestep',
    5: 'turnclap',
    # 6: 'numbersix',
    # 7: 'salute',
    # 8: 'mermaid',
    # 9: 'swing',
    # 10: 'cowboy',
    # 11: 'logout'
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
    accuracy = accuracy_score(Y_true, Y_pred)
    metrics = precision_recall_f1(Y_pred, Y_true, CLASSLIST)
    # micro_macro_weighted_scores = micro_macro_weighted(Y_pred, Y_true)
    cf_matrix = confusion_matrix(Y_true, Y_pred, labels=CLASSLIST)

    print("Results for " + dataset_type + " set...")
    print("Number of cases that were incorrect: " + str(num_incorrect))
    print("Accuracy: " + str(accuracy))

    for i in range(0, len(CLASSLIST)):
        # print("Precision " + CLASSLIST[i] + ": " + str(metrics[CLASSLIST[i]]['precision']))
        print("Recall " + CLASSLIST[i] + ": " + str(metrics[CLASSLIST[i]]['recall']))
        # print("F1 " + CLASSLIST[i] + ": " + str(metrics[CLASSLIST[i]]['f1']))

    # print("Micro precision: " + str(micro_macro_weighted_scores['micro_precision']))
    # print("Micro recall: " + str(micro_macro_weighted_scores['micro_recall']))
    # print("Micro f1: " + str(micro_macro_weighted_scores['micro_f1']))
    #
    # print("Macro precision: " + str(micro_macro_weighted_scores['macro_precision']))
    # print("Macro recall: " + str(micro_macro_weighted_scores['macro_recall']))
    # print("Macro f1: " + str(micro_macro_weighted_scores['macro_f1']))
    #
    # print("Weighted precision: " + str(micro_macro_weighted_scores['weighted_precision']))
    # print("Weighted recall: " + str(micro_macro_weighted_scores['weighted_recall']))
    # print("Weighted f1: " + str(micro_macro_weighted_scores['weighted_f1']))

    print("Confusion Matrix below " + str(CLASSLIST) + " : ")
    print(str(cf_matrix))

# Obtain a list of class probability values for every prediction
def recordClassProbabilites(pred):
    class_probabilities = []
    for i in range(0, len(pred)):
        prob_per_sentence = {}
        for j in range(0, len(pred[i])):
            prob_per_sentence[ENC_DICT[j]] = pred[i][j]
        class_probabilities.append(prob_per_sentence)
    return class_probabilities

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
    output = Dense(len(ENC_LIST), activation = 'softmax')(x)
    model = Model(inputs = main_input, outputs = output)
    # from keras import optimizers
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# Train the model, monitor on validation loss and save the best model out of given epochs
def fitModel(X_train, Y_train, X_val, Y_val):
    model = initialiseModel(X_train)
    filepath = os.path.join("nn_models", "nn_model" + "_{epoch:02d}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    sample_weight = compute_sample_weight('balanced', Y_train)
    model.fit(X_train, str2onehot(Y_train), epochs=20, validation_data=(X_val, str2onehot(Y_val)), batch_size=50, callbacks=callbacks_list, sample_weight=sample_weight)
    return model

def filterDataset(X, Y, X_test, Y_test):
    classes_removed = [
    # No classes need to be removed from self-collected dataset unless experimenting
        'NumberSix',
        'Salute',
        'Mermaid',
        'Swing',
        'Cowboy'
    ]

    del_idx = [ idx for idx, val in enumerate(Y) if val in classes_removed ]
    X = np.delete(X, del_idx, axis=0)
    Y = np.delete(Y, del_idx)

    del_idx = [ idx for idx, val in enumerate(Y_test) if val in classes_removed ]
    X_test = np.delete(X_test, del_idx, axis=0)
    Y_test = np.delete(Y_test, del_idx)

    return X, Y, X_test, Y_test

TRAIN_DATASET_PATH = os.path.join("dataset", "train.pkl")
TEST_DATASET_PATH = os.path.join("dataset", "test.pkl")
SCALER_SAVEPATH = os.path.join("scaler", "standard_scaler.pkl")
MODELS_SAVEPATH = os.path.join("nn_models")

if __name__ == "__main__":

    # scaler = QuantileTransformer(output_distribution='uniform')
    # scaler = MinMaxScaler((-1,1))
    scaler = StandardScaler()

    # Use the dataset prepared from self-collected dataset's raw data values
    X, Y = pickle.load(open(TRAIN_DATASET_PATH, 'rb'))
    X_test, Y_test = pickle.load(open(TEST_DATASET_PATH, 'rb'))
    X, Y, X_test, Y_test = filterDataset(X, Y, X_test, Y_test)

    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    pickle.dump(scaler, open(SCALER_SAVEPATH, 'wb'))

    print("Vectorizing...")

    # # Do some preprocess vectorizing for training/validation/testing sets respectively, as needed
    # vectorizer(...)
    # vectorizer(...)

    print("Fitting...")

    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42, shuffle=True, stratify=Y_test)

    print(str(Counter(Y)))
    print(str(Counter(Y_val)))
    print(str(Counter(Y_test)))

    model = fitModel(X, Y, X_val, Y_val)

    # Store only best model file and discard the rest
    modelFiles = os.listdir(MODELS_SAVEPATH)
    modelFiles.sort(reverse=True)
    for file in modelFiles[1:]:
        filepath = os.path.join(MODELS_SAVEPATH, file)
        if os.path.exists(filepath):
            os.remove(filepath)
    print("Renaming " + str(os.path.join(MODELS_SAVEPATH, modelFiles[0])) + " to " + str(os.path.join(MODELS_SAVEPATH, 'nn_model.hdf5')))
    os.rename(os.path.join(MODELS_SAVEPATH, modelFiles[0]), os.path.join(MODELS_SAVEPATH, 'nn_model.hdf5'))

    print("Predicting...")
    train_pred = model.predict(X)
    val_pred = model.predict(X_val)
    start = time.time()
    test_pred = model.predict(X_test)
    end = time.time()
    timeTaken = (end - start) / len(test_pred)
    print("Prediction time: " + str(timeTaken))

    print("Predictions done! Compiling results...")

    # Convert model output of class probabilities to corresponding best predictions
    Y_train_pred = onehot2str(train_pred)
    Y_val_pred = onehot2str(val_pred)
    Y_test_pred = onehot2str(test_pred)

    # Calculate accuracy, precision, recall and f1 scores
    calculatePerformanceMetrics(Y_train_pred, Y, "training")
    calculatePerformanceMetrics(Y_val_pred, Y_val, "validation")
    calculatePerformanceMetrics(Y_test_pred, Y_test, "testing")

    pred_prob = train_pred.tolist() + val_pred.tolist() + test_pred.tolist()
    pred_lbl = Y_train_pred.tolist() + Y_val_pred.tolist() + Y_test_pred.tolist()
    true_lbl = Y.tolist() + Y_val.tolist() + Y_test.tolist()
    for i in range(len(pred_prob)):
        pred_prob[i] = max(pred_prob[i])
    # Print model confidence for every wrong prediction
    printConfidenceForIncorrectPredictions(pred_prob, pred_lbl, true_lbl)
    # Print model confidence for every correct prediction
    printConfidenceForCorrectPredictions(pred_prob, pred_lbl, true_lbl)
