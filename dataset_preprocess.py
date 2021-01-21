import os
import numpy as np
import math
import configparser

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

def preprocess(path):

    files_path = []
    file_names = []
    files = os.listdir(path)
    for f in files:
        file_name = f.rstrip('.csv')
        file_names.append(file_name)

    rndseq = np.random.RandomState(config.getint('default', 'random_state')).permutation(file_names)

    if len(files) != 0:
        for j in range(len(files)):
            file_path = path + '/' + str(rndseq[j]) + '.csv'
            files_path.append(file_path)
            

    return files_path
