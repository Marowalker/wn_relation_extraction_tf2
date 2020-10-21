import os
import numpy as np

path = 'log'
f1_scores = []

for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    fn = open(filepath, 'r')
    lines = list(fn.readlines())
    last = lines[-1]
    comp = last.split(', ')
    f1 = float(comp[2])
    f1_scores.append(f1)

print("Mean F1 score: " + str(np.mean(f1_scores)))
