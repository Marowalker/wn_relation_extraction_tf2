import os
from nltk.corpus import wordnet as wn
from synset_file import numbers_only


datasets = ['train', 'dev', 'test']

input_path = 'data/raw_data'

# hype_dict = create_hype_dict()
depends = [d.strip() for d in open('data/all_depend.txt').readlines()]


def add_synsets(word):
    res = word
    w = word[:word.find('_')]
    if wn.synsets(w.lower()):
        replace = str(wn.synsets(w.lower())[0].offset())
    else:
        replace = str(0)
    if res.find('|') != -1:
        w1, p = res.split('\\')
        # print(w1)
        temp = p.split('|')
        temp.insert(1, replace)
        p = temp[0] + '\\' + '|'.join(temp[1:])
        # print(p)
        res = w1.lower() + '\\' + p
        # res = w1 + '|' + p
    # res = re.sub(reg, replace + "|", word)
    return res


def add_hypernyms(word):
    res = word
    w = word[:word.find('_')]
    if wn.synsets(w.lower()):
        if wn.synsets(w.lower())[0].hypernyms():
            replace = str(wn.synsets(w)[0].hypernyms()[0].offset())
        else:
            replace = str(0)
    else:
        replace = str(0)
    if res not in depends:
        w1, p = res.split('\\')
        # print(w1)
        temp = p.split('|')
        temp.insert(1, replace)
        p = temp[0] + '\\' + '|'.join(temp[1:])
        # print(p)
        res = w1.lower() + '\\' + p
        # res = w1 + '|' + p
    # res = re.sub(reg, replace + "|", word)
    return res


for dataset in datasets:
    print("Process dataset: " + dataset)
    with open(os.path.join(input_path, "sdp_data_acentors_graph." + dataset + ".txt"), 'r') as f:
        lines = f.readlines()
    with open(os.path.join(input_path, "sdp_data_acentors_graph_hypernyms." + dataset + ".txt"), 'w') as f2:
        for line in lines:
            if numbers_only(line):
                f2.write(line)
            else:
                temp = line.split()[:2]
                # print(temp)
                for token in line.split()[2:]:
                    # print(token)
                    # t = add_synsets(token)
                    t = add_hypernyms(token)
                    temp.append(t)
                sent = ' '.join(temp)
                f2.write(sent)
                f2.write('\n')


