from collections import defaultdict
from tqdm import tqdm
import csv


def get_sequence(data, k):
    n = len(data)
    dict_sequences = defaultdict(int)
    id_ = 0
    for x in tqdm(data):
        for j in range(n - k + 1):
            sequence = tuple(x[j : j + k])
            if sequence not in dict_sequences:
                dict_sequences[sequence] = id_
                id_ += 1
    return dict(dict_sequences)


def get_mismatch(all_sequences, data, k, n_mismatch):
    n = len(data)
    p = len(data[0])
    dict_sequences = {}
    embed = [{} for x in data]
    for i, x in enumerate(tqdm(data)):
        for j in range(p - k + 1):
            sequence = tuple(x[j : j + k])
            if sequence not in dict_sequences:
                neighbors = get_neighborhood(list(sequence), n_mismatch)
                dict_sequences[sequence] = [
                    all_sequences[tuple(neighbor)]
                    for neighbor in neighbors
                    if tuple(neighbor) in all_sequences
                ]
            for idx in dict_sequences[sequence]:
                if idx in embed[i]:
                    embed[i][idx] += 1
                else:
                    embed[i][idx] = 1
    return embed


def get_neighborhood(sequence, n_mismatches):
    m_list = [(0, "")]
    for letter in sequence:
        n_candidates = len(m_list)
        for i in range(n_candidates):
            mismatches, candidate = m_list.pop(0)
            if mismatches < n_mismatches:
                for adn_letter in "ATGC":
                    if adn_letter == letter:
                        m_list.append((mismatches, candidate + adn_letter))
                    else:
                        m_list.append((mismatches + 1, candidate + adn_letter))
            if mismatches == n_mismatches:
                m_list.append((mismatches, candidate + letter))
    return [candidate for mismatches, candidate in m_list]


def load_data():
    """
    function that loads data : X_train , y_train and X_test
    """
    # train features
    X_train = []

    with open("data/Xtr0.csv", "r") as file:
        X_train += [list(row[1]) for row in csv.reader(file)][1:]

    with open("data/Xtr1.csv", "r") as file:
        X_train += [list(row[1]) for row in csv.reader(file)][1:]

    with open("data/Xtr2.csv", "r") as file:
        X_train += [list(row[1]) for row in csv.reader(file)][1:]

    # train labels
    Y_train = []

    with open("data/Ytr0.csv", "r") as file:
        Y_train += [int(row[1]) for row in csv.reader(file) if row[1] != "Bound"]

    with open("data/Ytr1.csv", "r") as file:
        Y_train += [int(row[1]) for row in csv.reader(file) if row[1] != "Bound"]

    with open("data/Ytr2.csv", "r") as file:
        Y_train += [int(row[1]) for row in csv.reader(file) if row[1] != "Bound"]

    # test
    X_test = []

    with open("data/Xte0.csv", "r") as file:
        X_test += [list(row[1]) for row in csv.reader(file)][1:]

    with open("data/Xte1.csv", "r") as file:
        X_test += [list(row[1]) for row in csv.reader(file)][1:]

    with open("data/Xte2.csv", "r") as file:
        X_test += [list(row[1]) for row in csv.reader(file)][1:]

    return X_train, Y_train, X_test
