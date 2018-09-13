import pickle

RAW_DATA_PATH = "dummy_dataset\\RawData\\"

SAVE_FILEPATH = "dummy_dataset\\RawData_ByMove\\"

labels = []

ENC_DICT = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING',
    7: 'STAND_TO_SIT',
    8: 'SIT_TO_STAND',
    9: 'SIT_TO_LIE',
    10: 'LIE_TO_SIT',
    11: 'STAND_TO_LIE',
    12: 'LIE_TO_STAND'
}

with open(RAW_DATA_PATH + 'labels.txt') as labels_file:
    for line in labels_file:
        values = line.split(" ")
        experiment_no = values[0]
        if len(values[0]) == 1:
            experiment_no = "0" + values[0]
        user_no = values[1]
        if len(values[1]) == 1:
            user_no = "0" + values[1]
        move_id = int(values[2])
        move = ENC_DICT[move_id]
        start_line = int(values[3])
        end_line = int(values[4])
        labels.append((move_id, move, experiment_no, user_no, start_line, end_line))

# sort labels based on move and filter for just moves 1 to 6
# labels = [ x for x in labels if x[0] in range(1,7)]
labels = sorted(labels, key = lambda x: x[0])

moves = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']

data_by_move = {}
for move in moves:
    data_by_move[move] = []

for label in labels:
    acc_filepath = RAW_DATA_PATH + "acc_exp" + experiment_no + "_user" + user_no + ".txt"
    gyro_filepath = RAW_DATA_PATH + "gyro_exp" + experiment_no + "_user" + user_no + ".txt"

    data_acc = []
    data_gyro = []

    with open(acc_filepath) as acc_file:
        for i, line in enumerate(acc_file):
            if i in range(label[4], label[5]):
                data_acc.append(list(map(float, line.split(" "))))

    with open(gyro_filepath) as gyro_file:
        for i, line in enumerate(gyro_file):
            if i in range(label[4], label[5]):
                data_gyro.append(list(map(float, line.split(" "))))

    for i in range(len(data_acc)):
        data_acc[i].extend(data_gyro[i])

    data_by_move[label[1]].extend(data_acc)

for move in data_by_move:
    print(move + ": " + str(len(data_by_move[move])))

pickle.dump(data_by_move, open(SAVE_FILEPATH + 'data_by_move.pkl', 'wb'))
