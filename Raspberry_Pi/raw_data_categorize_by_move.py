import os, pickle

RAW_DATASET_PATH = os.path.join("dataset", "RawData")
SAVEPATH = os.path.join("dataset", "data_by_move.pkl")

moves = [ 'IDLE', 'logout', 'wipers', 'number7', 'chicken', 'sidestep', 'turnclap', 'numbersix', 'salute', 'mermaid', 'swing', 'cowboy' ]

data_by_move = {}

for move in moves:
    data_by_move[move] = []
    for dancer in os.listdir(RAW_DATASET_PATH):
        move_data_current_dancer = os.path.join(RAW_DATASET_PATH, dancer, move + '.txt')
        print(move_data_current_dancer)
        data_count = 0
        dancerDataAvailable = False
        if os.path.exists(move_data_current_dancer):
            with open(move_data_current_dancer) as textfile:
                for line in textfile:
                    values = line.split("\t")
                    if not len(values) == 9:
                        continue
                    values = [ val.strip().replace('\n', '') for val in values ]
                    values = list(map(float, values))
                    data_by_move[move].append(values)
                    data_count += 1
                    dancerDataAvailable = True
        if dancerDataAvailable == True:
            print("Recorded for " + move_data_current_dancer + " : " + str(data_count))

for move in data_by_move:
    print(move + ": " + str(len(data_by_move[move])))

pickle.dump(data_by_move, open(SAVEPATH, 'wb'))
