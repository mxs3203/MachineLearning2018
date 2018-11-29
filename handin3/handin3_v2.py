import math

import numpy as np
from collections import Counter
from pandas import *


def compute_border_array(input):
    n = len(input)
    ba = []

    ba.append(0)
    for i in range(1, n):
        b = ba[i - 1]
        while b > 0 and input[i] != input[b]:
            b = ba[b - 1]

        if input[i] == input[b]:
            ba.append(b + 1)
        else:
            ba.append(0)

    return ba


def ba_search(pattern, sequence):
    n = len(sequence)
    m = len(pattern)
    ba = compute_border_array(pattern + "$" + sequence)

    cnt = 0
    b = 0

    for i in range(0, len(ba)):
        if ba[i] == m:
            index = i - m + 1 - (m + 1)  # or i-2m
            print("ba_search: Report match on(1 indexed): ", index + 1)
            cnt = cnt + 1

    return cnt


def read_ann(filename):
    lines = []
    for l in open(filename).readlines():
        if l[0] != ">" and l[0] != ';':
            lines.append(l.strip())
    return "".join(lines)


def count_c(true, pred):
    total = tp = fp = tn = fn = 0
    for i in range(len(true)):
        if pred[i] == 'C' or pred[i] == 'c':
            total = total + 1
            if true[i] == 'C' or true[i] == 'c':
                tp = tp + 1
            else:
                fp = fp + 1
        if pred[i] == 'N' or pred[i] == 'n':
            if true[i] == 'N' or true[i] == 'n' or true[i] == 'R' or true[i] == 'r':
                tn = tn + 1
            else:
                fn = fn + 1
    return (total, tp, fp, tn, fn)


def count_r(true, pred):
    total = tp = fp = tn = fn = 0
    for i in range(len(true)):
        if pred[i] == 'R' or pred[i] == 'r':
            total = total + 1
            if true[i] == 'R' or true[i] == 'r':
                tp = tp + 1
            else:
                fp = fp + 1
        if pred[i] == 'N' or pred[i] == 'n':
            if true[i] == 'N' or true[i] == 'n' or true[i] == 'C' or true[i] == 'c':
                tn = tn + 1
            else:
                fn = fn + 1
    return (total, tp, fp, tn, fn)


def count_cr(true, pred):
    total = tp = fp = tn = fn = 0
    for i in range(len(true)):
        if pred[i] == 'C' or pred[i] == 'c' or pred[i] == 'R' or pred[i] == 'r':
            total = total + 1
            if (pred[i] == 'C' or pred[i] == 'c') and (true[i] == 'C' or true[i] == 'c'):
                tp = tp + 1
            elif (pred[i] == 'R' or pred[i] == 'r') and (true[i] == 'R' or true[i] == 'r'):
                tp = tp + 1
            else:
                fp = fp + 1
        if pred[i] == 'N' or pred[i] == 'n':
            if true[i] == 'N' or true[i] == 'n':
                tn = tn + 1
            else:
                fn = fn + 1
    return (total, tp, fp, tn, fn)


def print_stats(tp, fp, tn, fn):
    sn = float(tp) / (tp + fn)
    sp = float(tp) / (tp + fp)
    acp = 0.25 * (float(tp) / (tp + fn) + float(tp) / (tp + fp) + float(tn) / (tn + fp) + float(tn) / (tn + fn))
    ac = (acp - 0.5) * 2
    print("Sn = %.4f, Sp = %.4f, AC = %.4f" % (sn, sp, ac))


def print_all(true, pred):
    (totalc, tp, fp, tn, fn) = count_c(true, pred)
    if totalc > 0:
        print("Cs   (tp=%d, fp=%d, tn=%d, fn=%d):" % (tp, fp, tn, fn), end=" ")
        print_stats(tp, fp, tn, fn)

    (totalr, tp, fp, tn, fn) = count_r(true, pred)
    if totalr > 0:
        print("Rs   (tp=%d, fp=%d, tn=%d, fn=%d):" % (tp, fp, tn, fn), end=" ")
        print_stats(tp, fp, tn, fn)

    (total, tp, fp, tn, fn) = count_cr(true, pred)
    if totalc > 0 and totalr > 0:
        print("Both (tp=%d, fp=%d, tn=%d, fn=%d):" % (tp, fp, tn, fn), end=" ")
        print_stats(tp, fp, tn, fn)


def count_char(seq, from_char, to_char):
    count = 0
    for i in range(0, len(seq) - 1):
        if seq[i] == from_char and seq[i + 1] == to_char:
            count += 1
    return count

def create_count_matrix(ann):
    char_arr = ['C', 'R', 'N']
    count_mat = np.zeros(shape=(3, 3))

    for i in range(3):
        for n in range(3):
            count_mat[i, n] = count_char(ann, char_arr[i], char_arr[n])

    return count_mat


def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences


def chk_next(ann, i):
    if i == len(ann) - 1:
        return False
    if ann[i] == ann[i + 1]:
        return True
    return False


def extract_seq(seq, ann):
    i = 0
    start = 0
    end = 0
    n_list = []
    r_list = []
    c_list = []
    start_codons_c = []
    end_codons_c = []
    start_codons_r = []
    end_codons_r = []
    while i != len(ann):
        if ann[i] == 'N' and not chk_next(ann, i):
            end = i + 1
            n_list.append(seq[start:end])
            start = i + 1
        if ann[i] == 'C' and not chk_next(ann, i):
            end = i + 1
            if seq[start: start + 3] not in start_codons_c:
                start_codons_c.append(seq[start: start + 3])
            if seq[end - 3: end] not in end_codons_c:
                end_codons_c.append(seq[end - 3: end])
            c_list.append(seq[start + 3:end - 3])
            start = i + 1
        if ann[i] == 'R' and not chk_next(ann, i):
            end = i + 1
            if seq[start: start + 3] not in start_codons_r:
                start_codons_r.append(seq[start: start + 3])
            if seq[end - 3: end] not in end_codons_r:
                end_codons_r.append(seq[end - 3: end])
            r_list.append(seq[start + 3: end - 3])
            start = i + 1
        i += 1

    return c_list, r_list, n_list, start_codons_c, end_codons_c, start_codons_r, end_codons_r


def viterbi_test(trans_mat, emat, pi, seq):
    char_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    with np.errstate(divide='ignore'):
        pi = np.log(pi)
        trans_mat = np.log(trans_mat)
        emat = np.log(emat)

    lenSeq = len(seq)
    states = len(trans_mat)

    MaxArrow = np.zeros(shape=(states, lenSeq))

    MaxArrow.fill(float('-inf'))

    backtrack_seq = [0] * lenSeq
    score_matrix = np.zeros(shape=(states, lenSeq))
    score_matrix.fill(float('-inf'))

    score_matrix[:, 0] = pi + emat[char_dict[seq[0]],]

    for i in range(1, lenSeq):
        for x in range(states):
            prevCol = score_matrix[:, i - 1]
            em_prob = emat[char_dict[seq[i]], x]
            trans_prob = trans_mat[x, :]
            product = prevCol + trans_prob + [em_prob] * states

            score_matrix[x, i] = np.max(product)
            MaxArrow[x, i] = np.argmax(product)

    backtrack_seq[lenSeq - 1] = np.argmax(score_matrix[:, lenSeq - 1])

    for i in range(lenSeq - 1, 1, -1):
        # print(i)
        maxa = MaxArrow[backtrack_seq[i], i]
        # print(maxa)
        backtrack_seq[i - 1] = int(maxa)

    return backtrack_seq, score_matrix

def num_to_char(nums):
    dict = {0:'N', 1:'R', 2:'C'}
    output = []
    for i in nums:
        output.append(dict[i])
    return "".join(output).strip()


def cv(list_of_genAnn):
    counter = 1

    empty_model_trans = np.ones(shape=(3, 3))
    empty_model_emis = np.ones(shape=(3, 4))

    for validation_index in range(0, len(list_of_genAnn)):
        for train_index in range(0, len(list_of_genAnn)):
            if validation_index != train_index:
                print('train on: ', list_of_genAnn[train_index][0][0:10], "...", list_of_genAnn[train_index][1][0:10], "....")

                c_list, r_list, n_list, start_codons_c, end_codons_c, start_codons_r, end_codons_r = extract_seq(list_of_genAnn[train_index][0], list_of_genAnn[train_index][1])
                # so states are: c,r,n,sc,ec,sr,er

                # TODO: how to include start and stop codons in C and R?
                c_counts = [0, 0, 0, 0]
                r_counts = [0, 0, 0, 0]
                n_counts = [0, 0, 0, 0]

                for i in c_list:
                    c_counts[0] += i.count('A')
                    c_counts[1] += i.count('T')
                    c_counts[2] += i.count('C')
                    c_counts[3] += i.count('G')

                for i in r_list:
                    r_counts[0] += i.count('A')
                    r_counts[1] += i.count('T')
                    r_counts[2] += i.count('C')
                    r_counts[3] += i.count('G')

                for i in n_list:
                    n_counts[0] += i.count('A')
                    n_counts[1] += i.count('T')
                    n_counts[2] += i.count('C')
                    n_counts[3] += i.count('G')

                empty_model_emis += [c_counts, r_counts, n_counts]
                empty_model_trans += create_count_matrix(list_of_genAnn[train_index][1])




        # updating the empty model with new counts
        empty_model_trans = np.divide(empty_model_trans, empty_model_trans.sum(axis=1, keepdims=True),
                                      out=np.zeros_like(empty_model_trans),
                                      where=empty_model_trans.sum(axis=1, keepdims=True) != 0)
        empty_model_emis = np.divide(empty_model_emis,
                                     empty_model_emis.sum(axis=0, keepdims=True),
                                     out=np.zeros_like(empty_model_emis),
                                     where=empty_model_emis.sum(axis=0, keepdims=True) != 0)
        print(empty_model_emis)
        print(empty_model_trans)

        print('validate on:', list_of_genAnn[validation_index][1][0:10], "...")
        viterbi_seq = viterbi_test(empty_model_trans, empty_model_emis.transpose(), [1,0,0], list_of_genAnn[validation_index][0])
        viterbi_seq = num_to_char(viterbi_seq[0])
        print_all(list_of_genAnn[validation_index][1], viterbi_seq)

    return empty_model_trans, empty_model_emis


def predict(unknown, trans_mat, emis_mat):
    counter = 6
    for genome in unknown:
        results = viterbi_test(trans_mat,emis_mat, [1,0,0], genome)
        viterbi_seq = num_to_char(results[0])
        decoding_gen = open("decoding_gen" + str(counter) + '.fa', 'x')
        decoding_gen.write("> pred-ann" + str(counter) + "\n" + viterbi_seq)
        counter += 1
    return -1


if __name__ == '__main__':
    char_arr = ['C', 'R', 'N']
    ################# Unit Testing ####################################

    full_path = 'C:/Users/Ky/Desktop/ml18/handin3/'
    full_path = ''
    gen1 = read_fasta_file(full_path + 'data-handin3/genome1.fa')
    ann1 = read_fasta_file(full_path + 'data-handin3/true-ann1.fa')

    gen2 = read_fasta_file(full_path + 'data-handin3/genome2.fa')
    ann2 = read_fasta_file(full_path + 'data-handin3/true-ann2.fa')

    gen3 = read_fasta_file(full_path + 'data-handin3/genome3.fa')
    ann3 = read_fasta_file(full_path + 'data-handin3/true-ann3.fa')

    gen4 = read_fasta_file(full_path + 'data-handin3/genome4.fa')
    ann4 = read_fasta_file(full_path + 'data-handin3/true-ann4.fa')

    gen5 = read_fasta_file(full_path + 'data-handin3/genome5.fa')
    ann5 = read_fasta_file(full_path + 'data-handin3/true-ann5.fa')

    gen6 = read_fasta_file(full_path + 'data-handin3/genome6.fa')
    gen7 = read_fasta_file(full_path + 'data-handin3/genome7.fa')
    gen8 = read_fasta_file(full_path + 'data-handin3/genome8.fa')
    gen9 = read_fasta_file(full_path + 'data-handin3/genome9.fa')
    gen10 = read_fasta_file(full_path + 'data-handin3/genome10.fa')

    gen_arr = [[gen1['genome1'], ann1['true-ann1']], [gen2['genome2'], ann2['true-ann2']],
               [gen3['genome3'], ann3['true-ann3']], [gen4['genome4'], ann4['true-ann4']],
               [gen5['genome5'], ann5['true-ann5']]]

    unknown_arr = [gen6['genome6'], gen7['genome7'], gen8['genome8'], gen9['genome9'], gen10['genome10']]

    # print(list(gen_arr[0][0].items())[0][0])
    cv_trans_mat, cv_emis_mat = cv(gen_arr)
    predict(unknown_arr, cv_trans_mat, cv_emis_mat)
