import os, pickle

RAW_DATASET_PATH = "dataset\\RawData\\"
SAVEPATH = "dataset\\"

moves = ['idle', 'logout', 'number_six']

data_by_move = {}

for move in moves:
    data_by_move[move] = []
    for dancer in os.listdir(RAW_DATASET_PATH):
        move_data_current_dancer = RAW_DATASET_PATH + dancer + '\\' + move + '.txt'
        if os.path.exists(move_data_current_dancer):
            with open(move_data_current_dancer) as textfile:
                for line in textfile:
                    values = line.split("\t")
                    values = list(map(float, values))
                    data_by_move[move].append(values)

for move in data_by_move:
    print(move + ": " + str(len(data_by_move[move])))

pickle.dump(data_by_move, open(SAVEPATH + 'data_by_move.pkl', 'wb'))
