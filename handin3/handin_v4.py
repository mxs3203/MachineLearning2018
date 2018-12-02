import numpy as np
import math

import itertools


######################################### COMPARE ANN ####################################################

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


####################### END OF COMPARE ANN ##################################################


def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)


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

def splitStringByCodon(seq):
    list = []
    for i in range(0, len(seq), 3):
        if len(seq[i:i + 3]) == 3:
            list.append(seq[i:i + 3])

    return list


def viterbi(init_probs, trans_probs, emission_probs, seq):

    codonToIndex = mapCodonsToIndexes()
    seqCodons = splitStringByCodon(seq)

    k = len(init_probs)
    n = len(seqCodons)

    w = [[0] * n for _ in range(k)]

    for i in range(k):
        w[i][0] = log(init_probs[i]) + log(emission_probs[i][codonToIndex[seqCodons[0]]])

    for j in range(1, n):
        for i in range(k):
            w[i][j] = max([log(emission_probs[i][codonToIndex[seqCodons[j]]]) + w[g][j - 1] + log(trans_probs[g][i]) for g in range(k)])

    N = len(w[0])
    K = len(w)

    # p = max(w[i][-1] for i in range(len(w)))
    backtrack = [0 for k in range(N)]

    backtrack[N - 1] = max(range(K), key=lambda k: w[k][N - 1])

    for i in range(N - 2, -1, -1):
        backtrack[i] = max(range(K), key=lambda k:
        log(emission_probs[backtrack[i + 1]][codonToIndex[seqCodons[i + 1]]]) + w[k][i] + log(trans_probs[k][backtrack[i + 1]]))

    new_backtrack = []
    # multiply every state by 3 because we are doing it for codons and we need per char
    for state in backtrack:
        for i in range(3):
            new_backtrack.append(state)
    return new_backtrack


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


def updateTransMatrix(old_trans, trans):
    for i in range(0, len(old_trans)):
        for j in range(0, len(old_trans)):
            old_trans[i][j] = old_trans[i][j] + trans[i][j]

    return old_trans


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
            start_codons_c.append(seq[start: start + 3])
            end_codons_c.append(seq[end - 3: end])
            c_list.append(seq[start + 3:end - 3])
            start = i + 1
        if ann[i] == 'R' and not chk_next(ann, i):
            end = i + 1
            start_codons_r.append(seq[start: start + 3])
            end_codons_r.append(seq[end - 3: end])
            r_list.append(seq[start + 3: end - 3])
            start = i + 1
        i += 1

    return c_list, r_list, n_list, start_codons_c, end_codons_c, start_codons_r, end_codons_r


def allPossibleBases(size):
    bases = ['A', 'C', 'T', 'G']

    perms = [''.join(p) for p in itertools.product(bases, repeat=size)]
    dict = {}
    for i in perms:
        dict[i] = 1
    return dict


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def countCodons(list, all_possible):
    for seq in list:
        for i in range(0, len(seq), 3):
            if len(seq[i:i + 3]) == 3:
                all_possible[seq[i:i + 3]] += 1

    return all_possible


def tranfromListToDict(list):
    dict = {}
    for i in list:
        dict[i] = 0
    for item in dict:
        for i in list:
            if item == i:
                dict[item] += 1
    return dict


# function replaces 3 states sequences with 7 states sequence which is used for modeling
# Or it can do the opposite (7states to 3states), which is used for comparison to true seq(not modeling)
def replaceAnnotatedSeq(ann, forModeling=True):  # for modeling will return states we use for modeling if true
    # if false, it will do the opposite, transform it back to original 3 state sequence
    new_ann = list(ann)  # make it a list of chars so I can replace specific positions(with string you cant)
    if forModeling:
        for i in range(0, len(ann) - 1):
            if ann[i] == 'C' and ann[i + 1] == 'N':  # end C = y
                new_ann[i + 1] = 'Y'
            if ann[i] == 'N' and ann[i + 1] == 'C':  # Start C = x
                new_ann[i + 1] = 'X'
            if ann[i] == 'N' and ann[i + 1] == 'R':  # start R = z
                new_ann[i + 1] = 'Z'
            if ann[i] == 'R' and ann[i + 1] == 'N':  # End R = W
                new_ann[i + 1] = 'W'
    else:
        for i in range(1, len(ann) - 1):
            if ann[i] == 'Y':  # end C
                new_ann[i - 1] = 'C'
            if ann[i] == 'X':  # start C
                new_ann[i - 1] = 'N'
            if ann[i] == 'Z':  # start R
                new_ann[i - 1] = 'N'
            if ann[i] == 'W':  # end R
                new_ann[i - 1] = 'R'

    return "".join(new_ann)


def create_count_matrix(ann):
    #                           Start C, end C, start R, end R, C to R = K, R to C = L
    char_arr = ['C', 'R', 'N', 'X', 'Y', 'Z', 'W']
    count_mat = np.zeros(shape=(7, 7))

    for i in range(7):
        for n in range(7):
            count_mat[i, n] = count_char(ann, char_arr[i], char_arr[n])

    return count_mat


def translateObs_toindex(obs):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [mapping[symbol.upper()] for symbol in obs]


# The function will transform sequence to a indexes
def transformAnnToIndexes(ann):
    dict = {'C': 0, 'R': 1, 'N': 2, 'X': 3, 'Y': 4, 'Z': 5, 'W': 6}

    return [dict[symbol.upper()] for symbol in ann.upper()]


# The function will transform sequence to a indexes
def transformIndexesToAnnSeq(ann):
    dict = {0: 'C', 1: 'R', 2: 'N', 3: 'X', 4: 'Y', 5: 'Z', 6: 'W'}

    return [dict[symbol] for symbol in ann]


def count_char(seq, from_char, to_char):
    count = 0
    for i in range(0, len(seq) - 1):
        if seq[i] == from_char and seq[i + 1] == to_char:
            count += 1
    return count

def mapCodonsToIndexes():
    codons = allPossibleBases(size=3)
    map = {}
    cnt = 0
    for codon in codons:
        map[codon] = cnt
        cnt = cnt + 1

    return map



if __name__ == '__main__':

    # ################# Unit Testing ####################################

    full_path = 'C:/Users/Ky/Desktop/ml18/handin3/'
    full_path = 'handin3/'
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

    emiss_count = np.zeros((7, len(allPossibleBases(size=3))))
    trans_count = np.ones((7, 7))
    for i in gen_arr:

        c_list, r_list, n_list, start_codons_c, end_codons_c, start_codons_r, end_codons_r = extract_seq(i[0], i[1])
        # Emission matrix
        dict_of_all_codons_c = allPossibleBases(size=3)
        dict_of_all_codons_r = allPossibleBases(size=3)
        dict_of_all_codons_n = allPossibleBases(size=3)

        dict_of_all_start_c = tranfromListToDict(start_codons_c)
        dict_of_all_end_c = tranfromListToDict(end_codons_c)
        dict_of_all_start_r = tranfromListToDict(start_codons_r)
        dict_of_all_end_r = tranfromListToDict(end_codons_r)

        dict_of_all_codons_c = countCodons(c_list, dict_of_all_codons_c)
        dict_of_all_codons_r = countCodons(r_list, dict_of_all_codons_r)
        dict_of_all_codons_n = countCodons(n_list, dict_of_all_codons_n)

        j = 0
        for key in dict_of_all_codons_c.keys():
            emiss_count[0][j] = emiss_count[0][j] + dict_of_all_codons_c[key]
            emiss_count[1][j] = emiss_count[1][j] + dict_of_all_codons_r[key]
            emiss_count[2][j] = emiss_count[2][j] + dict_of_all_codons_n[key]
            if key in dict_of_all_start_c.keys():
                emiss_count[3][j] = emiss_count[3][j] + dict_of_all_start_c[key]
            if key in dict_of_all_end_c.keys():
                emiss_count[4][j] = emiss_count[4][j] + dict_of_all_end_c[key]
            if key in dict_of_all_start_r.keys():
                emiss_count[5][j] = emiss_count[5][j] + dict_of_all_start_r[key]
            if key in dict_of_all_end_r.keys():
                emiss_count[6][j] = emiss_count[6][j] + dict_of_all_end_r[key]
            j = j + 1

        ann_with_intermedian_states = replaceAnnotatedSeq(i[1], forModeling=True)
        trans_matrix = create_count_matrix(ann_with_intermedian_states)
        trans_count = updateTransMatrix(trans_count, trans_matrix)

    emis_prob_matrix = np.divide(emiss_count,
                                 emiss_count.sum(axis=1, keepdims=True),
                                 out=np.zeros_like(emiss_count),
                                 where=emiss_count.sum(axis=1, keepdims=True) != 0)
    trans_prob_matrix = np.divide(trans_count,
                                  trans_count.sum(axis=1, keepdims=True),
                                  out=np.zeros_like(trans_count),
                                  where=trans_count.sum(axis=1, keepdims=True) != 0)
    print("C             R           N           SC(X)         EC(Y)         SR(z)          ER(W) ")
    matprint(trans_prob_matrix)
    # start with N
    init = [0, 0, 1, 0, 0, 0, 0]
    # Let try to get results for first annotated genome...
    raw_viterbi = viterbi(init_probs=init,
                          trans_probs=trans_prob_matrix,
                          emission_probs=emis_prob_matrix,
                          seq=gen_arr[0][0])
    print(len(raw_viterbi))
    print(len(gen_arr[0][1][0:len(gen_arr[0][1])-1]))
    #print(raw_viterbi)
    # vterbi returns numbers corresponding to 7 states

    statesSeq = transformIndexesToAnnSeq(raw_viterbi)  # transform indexes to states(chars)
    #print(statesSeq)
    decodedGenome = replaceAnnotatedSeq(statesSeq, forModeling=False)  # for modeling is with 7 states(chars) ,without with 3
    #print(decodedGenome)

    decoding_gen = open("decoding_gen" + str(1) + '.fa', 'x')
    decoding_gen.write("> pred-ann" + str(1) + "\n" + str(raw_viterbi) + "\n" + str(statesSeq) + "\n" + decodedGenome)
    print_all(gen_arr[0][1][0:len(gen_arr[0][1])-1], decodedGenome)
