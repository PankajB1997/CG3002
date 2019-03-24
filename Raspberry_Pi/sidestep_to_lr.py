import os, pickle
import random

RAW_DATASET_PATH = os.path.join("dataset", "RawData")
SAVEPATH = os.path.join("dataset", "data_by_move.pkl")
RES_SAVEPATH = os.path.join("dataset", "left_right.pkl")

def assignCorrectly(result):
    return result

moves = [ 'sidestep' ]
res_moves = [ 'left', 'right' ]
choice = {
    "bryan": 0, 
    "pankaj": 1, 
    "jinting": 0, 
    "jin": 1, 
    "junyang": 0, 
    "yanyu": 0
}

result = { 'left': [], 'right': [] }

data_by_move = {}

for move in moves:
    data_by_move[move] = []
    for dancer in os.listdir(RAW_DATASET_PATH):
        if dancer == "jin":
            continue
        move_data_current_dancer = os.path.join(RAW_DATASET_PATH, dancer, move + '.txt')
        print(move_data_current_dancer)
        data_count = 0
        left_right_flag = choice[dancer]
        dancerDataAvailable = False
        if os.path.exists(move_data_current_dancer):
            with open(move_data_current_dancer) as textfile:
                for i,line in enumerate(textfile):
                    values = line.split("\t")
                    if not len(values) == 9:
                        continue
                    values = values[:6]
                    values = [ val.strip().replace('\n', '') for val in values ]
                    values = list(map(float, values))
                    result[res_moves[left_right_flag]].append(values)
                    if i % 32 == 0:
                        left_right_flag = 1 - left_right_flag
                    data_by_move[move].append(values)
                    data_count += 1
                    dancerDataAvailable = True
        if dancerDataAvailable == True:
            print("Recorded for " + move_data_current_dancer + " : " + str(data_count))

for move in data_by_move:
    print(move + " : " + str(len(data_by_move[move])))

pickle.dump(data_by_move, open(SAVEPATH, 'wb'))
result = assignCorrectly(result)
print(len(result['left']))
print(len(result['right']))
pickle.dump(result, open(RES_SAVEPATH, 'wb'))