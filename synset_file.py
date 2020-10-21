from nltk.corpus import wordnet as wn
import re
import os
from collections import defaultdict


datasets = ['train', 'dev', 'test']

input_path = 'data/raw_data'

all_tokens = []


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''


def numbers_only(sent):
    m = re.match(r'.*[A-z]+', sent)
    if m:
        return False
    return True


def check_parentheses(line):
    reg1 = r'\(.*'
    reg2 = r'.*\)'
    # res = re.sub(reg, ' ', line)
    m1 = re.match(reg1, line)
    m2 = re.match(reg2, line)
    if m1 or m2:
        return True
    return False


def get_max_offset():
    res = 0
    for ss in wn.all_synsets():
        if res < ss.offset():
            res = ss.offset()
    return res


def check_hypernyms(synsets):
    for s in synsets:
        if s.hypernyms():
            return s.hypernyms()[0]
    return None


def create_all_set_lists():
    all_sets = []
    hype = []
    no_hype = []
    for dataset in datasets:
        # print("Process dataset: " + dataset)
        with open(os.path.join(input_path, "sdp_data_acentors." + dataset + ".txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if numbers_only(line):
                    # f2.write(line)
                    pass
                else:
                    # temp = line.split()[:2]
                    # print(temp)
                    for token in line.split()[2:]:
                        # print(token)
                        w = token[:token.find('_')]
                        # print(w)
                        if not check_parentheses(w):
                            if w.lower() not in all_tokens:
                                all_tokens.append(w.lower())
                            if wn.synsets(w.lower()):
                                # print(w, "synset")
                                if w.lower() not in all_sets:
                                    all_sets.append(w.lower())
                                if check_hypernyms(wn.synsets(w.lower())):
                                    # print(w, "hype")
                                    # print(wn.synsets(w.lower())[0].hypernyms()[0])
                                    if w.lower() not in hype:
                                        hype.append(w.lower())
                                else:
                                    # print(w)
                                    if w.lower() not in no_hype:
                                        no_hype.append(w.lower())
    return all_sets, hype, no_hype


temp = []
all_sets, hype, no_hype = create_all_set_lists()


def create_hype_dict():
    res = defaultdict()
    for t in sorted(hype, key=str.lower):
        offset = check_hypernyms(wn.synsets(t)).offset()
        res[t] = offset
    return res


# d = create_hype_dict()
#
# for i in d:
#     if d[i] not in temp:
#         temp.append(d[i])
#
# with open('data/all_hypernyms.txt', 'w') as f:
#     for token in temp:
#         f.write(str(token))
#         f.write('\n')
#     f.close()








