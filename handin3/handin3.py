# http://users-cs.au.dk/cstorm/courses/ML_e18/projects/handin3/ml-handin-3.html
import math

import numpy as np
from collections import Counter
from pandas import *

TESTING = False


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


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


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


def make_trans(count_mat):
    trans_mat = np.zeros(shape=(3, 3))
    for i in range(0, 3):
        for x in range(0, 3):
            trans_mat[i, x] = count_mat[i, x] / sum(count_mat[i])
    return trans_mat


def chk_next(ann, i):
    if i == len(ann) - 1:
        return False
    if ann[i] == ann[i + 1]:
        return True
    return False


# TODO:
def make_reverse_complement(seq):
    seq = seq.upper()
    print(seq)
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

    # "".join(complement[base] for base in seq)
    # reversed or not reveresed ?
    return "".join(complement[base] for base in reversed(seq))


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


def translate_path_to_indices_3state(obs):
    mapping = {"N": 0, "n": 0, "C": 1, "c": 1, "R": 2, "r": 2}
    return [mapping[symbol.lower()] for symbol in obs]


def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]


def make_emission_count(seq, ann):
    N = len(seq)
    indeces_ann = translate_path_to_indices_3state(ann)
    indeces_seq = translate_observations_to_indices(seq)
    matrix_emission = np.zeros(shape=(3, 4))  # 3 states(N,C,R) and 4 emisssions(ACGT)

    for n in range(N):
        matrix_emission[indeces_ann[n]][indeces_seq[n]] += 1

    return matrix_emission


def make_emission_prob_matrix(count_matrix_emission):
    emission_matrix = np.zeros(shape=(3, 4))
    for i in range(0, 3):
        for x in range(0, 4):
            emission_matrix[i, x] = count_matrix_emission[i, x] / sum(count_matrix_emission[i])
    return emission_matrix


def ky_make_emission_count(seq, ann):
    c_list, r_list, n_list, c_start_codons, c_stop_codons, r_start_codons, r_stop_codons = extract_seq(seq, ann)

    #           ATCG
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

    for i in range(0, len(c_counts)):
        length = sum(c_counts)
        c_counts[i] = c_counts[i] / length

    for i in range(0, len(r_counts)):
        length = sum(r_counts)
        r_counts[i] = r_counts[i] / length

    for i in range(0, len(n_counts)):
        length = sum(n_counts)
        n_counts[i] = n_counts[i] / length

    return Counter(c_start_codons), Counter(r_start_codons), c_counts, r_counts, n_counts


def make_hmm(old_trans, old_emat):
    new_trans = np.zeros(shape=(15, 15))
    for i in range(14):
        new_trans[i, i + 1] = 1
    new_trans[0, 0] = old_trans[2, 2]
    new_trans[7, 0] = old_trans[0, 2]
    new_trans[14, 0] = old_trans[1, 2]
    new_trans[14, 1] = old_trans[1, 0]
    new_trans[0, 1] = old_trans[2, 0]
    new_trans[4, 4] = old_trans[0, 0]
    new_trans[4, 5] = old_trans[0, 1] + old_trans[0, 2]
    new_trans[0, 8] = old_trans[2, 1]
    new_trans[7, 8] = old_trans[0, 1]
    new_trans[11, 11] = old_trans[1, 1]
    new_trans[11, 12] = old_trans[1, 0] + old_trans[1, 2]

    new_emat = np.zeros(shape=(4, 15))
    new_emat[0,] = [0, 1] + [0] * 4 + [1, 1] + [0] * 2 + [1] + [0] * 2 + [1, 0]
    new_emat[1,] = [0] * 2 + [1] + [0] * 2 + [1] + [0] * 2 + [1, 1] + [0] * 4 + [1]
    new_emat[2,] = [0] * 12 + [1] + [0, 0]
    new_emat[3,] = [0] * 3 + [1] + [0] * 11
    new_emat[:, 0] = old_emat[:, 0]
    new_emat[:, 4] = old_emat[:, 1]
    new_emat[:, 11] = old_emat[:, 2]

    return new_trans, new_emat


def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x
                    )


def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]


def make_log(trans_matrix, emission_matrix, init_prob):
    K = len(init_prob)
    # Create empty matrices for filling in
    emission_probs = make_table(K, len(emission_matrix[0]))
    trans_probs = make_table(K, K)

    # PUT THEM on LOG scale
    init_prob_log = [log(y) for y in init_prob]

    # emission
    for i in range(K):
        for j in range(len(emission_matrix[i])):
            emission_probs[i][j] = log(emission_matrix[i][j])
    # transition
    for i in range(K):
        for j in range(K):
            trans_probs[i][j] = log(trans_matrix[i][j])

    return trans_probs, emission_probs, init_prob_log


def num_to_char(nums):
    # S = c_start, E = c_end, Z = r_start, F = r_end
    dict = {0 : 'N',1: 3*'something'}
    dict = ['N'] + ['S'] * 3 + ['C'] + ['E'] * 3 + ['Z'] * 3 + ['R'] + ['F'] * 3
    output = []
    for i in nums:
        output.append(dict[i])
    return "".join(output).strip()


def num_to_char_3state(nums):
    # S = c_start, E = c_end, Z = r_start, F = r_end
    dict = ['N'] + ['C'] * 3 + ['C'] + ['C'] * 3 + ['R'] * 3 + ['R'] + ['R'] * 3
    print(dict)
    output = []
    for i in nums:
        output.append(dict[i])
    return "".join(output).strip()


def viterbi(trans_matrix, emission_matrix, init_prob, seq):
    k = len(init_prob)
    n = len(seq)

    w = [[0] * n for _ in range(k)]

    for i in range(k):
        w[i][0] = log(init_prob[i]) + log(emission_matrix[i][seq[0]])

    for j in range(1, n):
        for i in range(k):
            w[i][j] = max(
                [log(emission_matrix[i][seq[j]]) + w[g][j - 1] + log(trans_matrix[g][i]) for g in range(k)])

    maxx = max(w[i][-1] for i in range(len(w)))

    N = len(w[0])
    K = len(w)

    z = [0 for k in range(N)]

    z[N - 1] = max(range(K), key=lambda k: w[k][N - 1])

    for i in range(N - 2, -1, -1):
        z[i] = max(range(K), key=lambda k:
        log(emission_matrix[z[i + 1]][seq[i + 1]]) + w[k][i] + log(trans_matrix[k][z[i + 1]]))

    return z


def print_matrix(matrix):
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))


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


def cv(list_of_genAnn):
    counter = 1
    empty_model_trans = np.ones(shape=(3, 3))
    empty_model_emis = np.ones(shape=(3, 4))

    for i in list_of_genAnn:

        for j in list_of_genAnn:
            if i != j:
                print('train on: ', j[0][0:10], "...", j[1][0:10], "....")
                # TODO: count all possible states and emission and append it to empty model trans of emis
                # TODO: instead of counting 3x4 and 4x4 we have to count for all 15 x15 matrix and 4x15

                empty_model_trans = empty_model_trans + create_count_matrix(j[1])  # ann
                empty_model_emis = empty_model_emis + make_emission_count(j[0], j[1])  # gen. ann

        # updating the empty model with new counts
        empty_model_trans = np.divide(empty_model_trans, empty_model_trans.sum(axis=1, keepdims=True),
                                      out=np.zeros_like(empty_model_trans),
                                      where=empty_model_trans.sum(axis=1, keepdims=True) != 0)
        empty_model_emis = np.divide(empty_model_emis,
                                     empty_model_emis.sum(axis=1, keepdims=True),
                                     out=np.zeros_like(empty_model_emis),
                                     where=empty_model_emis.sum(axis=1, keepdims=True) != 0)
        print('validate on:', i[1][0:10], "...")

        trans_mat = make_trans(empty_model_trans)
        em_mat = make_emission_prob_matrix(empty_model_emis)

        hmm = make_hmm(trans_mat, em_mat.transpose())
        results = viterbi_test(hmm[0], hmm[1], [1] + [0] * 14, i[0])
        viterbi_seq = num_to_char_3state(results[0])

        # decoding_gen = open("decoding_gen" + str(counter) + '.fa', 'x')
        # decoding_gen.write("> pred-ann" + str(counter) + "\n" + viterbi_seq)
        print_all(i[1], viterbi_seq)
        counter = counter + 1

    return empty_model_trans, empty_model_emis


def predict(unknown, trans_mat, emis_mat, cv_init_prob):
    counter = 6
    for genome in unknown:
        hmm = make_hmm(trans_mat, emis_mat.transpose())
        results = viterbi_test(hmm[0], hmm[1], [1] + [0] * 14, genome)
        viterbi_seq = num_to_char_3state(results[0])
        decoding_gen = open("decoding_gen" + str(counter) + '.fa', 'x')
        decoding_gen.write("> pred-ann" + str(counter) + "\n" + viterbi_seq)
        counter += 1


def test_exstract_seq(seq, ann):
    print('-------------Starting Test  extract Seq with:------------- ')
    print(seq)
    print(ann)
    print('Extracting sequences based on annotation: ')
    result = extract_seq(seq, ann)
    print("C: ", result[0])
    print("R: ", result[1])
    print("N: ", result[2])
    print("C_start:", result[3])
    print("C_end", result[4])
    print("R_start:", result[5])
    print("R_end", result[6])
    assert result[0] == ['GT']
    assert result[1] == ['GT']
    assert result[2] == ['CTGACGTCAGC', 'ACA', 'C']
    assert result[3] == ['TAC']
    assert result[4] == ['ACG']
    assert result[5] == ['TGC']
    assert result[6] == ['ATT']

    print('-------------Test Extracting sequences Success!------------- \n\n')


def test_count_mat(gen_arr, char_arr):
    print('-------------Starting Test Count Matrix with: -------------')
    print(gen_arr)
    print(char_arr)
    print('Creating count matrix based on observations: ')

    result = create_count_matrix(gen_arr, char_arr)
    expected = [[11., 1., 0.],
                [0., 11., 1.],
                [1., 0., 11.]]
    print(result)
    # assert result[0][0] == expected[0][0], result[0][1] == expected[0][1]
    # assert result[0][2] == expected[0][2], result[1][0] == expected[1][0]
    # assert result[2][0] == expected[2][0], result[2][1] == expected[2][1]
    print("-------------Test Count Matrix success! -------------\n\n")

    return result


def test_make_trans(count_mat):
    print('-------------Starting Trans Prob Matrix with: -------------')
    print(count_mat)
    print('Creating trans prob matrix based on count matrix: ')

    result = make_trans(count_mat)
    expected = [[11. / 12, 1 / 12., 0.],
                [0., 11 / 12., 1. / 12],
                [1. / 12, 0., 11. / 12]]
    # assert result[0][0] == expected[0][0], result[0][1] == expected[0][1]
    # assert result[0][2] == expected[0][2], result[1][0] == expected[1][0]
    # assert result[2][0] == expected[2][0], result[2][1] == expected[2][1]
    print("-------------Test Trans Prob Matrix success! -------------\n\n")


def test_count_char(seq, from_char, to_char, exp):
    print('-------------Starting count char with: -------------')
    print(seq)
    print(from_char)
    print(to_char)
    print('Counting chars, from - to: ')

    result = count_char(seq, from_char, to_char)
    print(result)
    assert result == exp

    print("------------- Test Count Char success! -------------\n\n")


def test_emis_matrix(seq, ann):
    print('-------------Starting Emission matrix Matrix: -------------')

    print(seq)
    print(ann)
    print("rows = N C R")
    print("cols = A C G T")
    emis_count = make_emission_count(seq=seq,
                                     ann=ann)

    print(emis_count)

    print(make_emission_prob_matrix(emis_count))

    assert emis_count[0][0] == 4
    assert emis_count[0][1] == 5
    assert emis_count[0][2] == 3
    assert emis_count[0][3] == 3

    assert emis_count[1][0] == 2
    assert emis_count[1][1] == 2
    assert emis_count[1][2] == 2
    assert emis_count[1][3] == 3

    print("------------- Test Emis matrix success! -------------\n\n")


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

    # c_list, r_list, n_list, start_codons_c, end_codons_c, start_codons_r, end_codons_r

    # trans_list =  []
    # em_list = []
    #
    # for n in range(5):
    #
    #     c_list, r_list, n_list, c_start, c_end, r_start, r_end = [], [], [], [], [], [], []
    #     trans_counts = np.zeros(shape=(3, 3))
    #     emission_counts = np.zeros(shape=(3, 4))
    #
    #     for i in [x for x in range(5) if x not in [n]]:
    #         seq, ann = list(gen_arr[i][0].items())[0][1], list(gen_arr[i][1].items())[0][1]
    #         result = extract_seq(seq, ann)
    #         c_list.append(result[0])
    #         r_list.append(result[1])
    #         n_list.append(result[2])
    #         c_start.append(result[3])
    #         c_end.append(result[4])
    #         r_start.append(result[5])
    #         r_end.append(result[6])
    #
    #         trans_counts += create_count_matrix(seq, ann)
    #         emission_counts += make_emission_count(seq, ann)
    #
    #     trans_mat = make_trans(trans_counts)
    #     em_mat = make_emission_prob_matrix(emission_counts)
    #
    #     trans_list.append(trans_mat)
    #     em_list.append(em_mat)
    #
    #     hmm = make_hmm(trans_mat, em_mat.transpose())
    #     results = viterbi_test(hmm[0], hmm[1], [1]+[0]*14, list(list(gen_arr[n][0].items())[0][1]))
    #
    #     decoding_gen = open("decoding_gen" + str(n), 'x')
    #     decoding_gen.write(''.join(num_to_char_3state(results[0])))
    #
    # trans_mat = sum(trans_list)/5
    # em_mat = sum(em_list)/5
    # hmm = make_hmm(trans_mat, em_mat.transpose())
    #
    # for i in range(6,11):
    #     results = viterbi_test(hmm[0], hmm[1],[1]+[0]*14,list(unknown_arr[i-6].items())[0][1])
    # #print(''.join(num_to_char(results[0])))
    #     decoding_gen = open("decoding_gen"+str(i),'x')
    #     decoding_gen.write(''.join(num_to_char_3state(results[0])))
    #
    #
    # np.savetxt('trans_mat',hmm[0],delimiter=',',fmt='%1.3f')
    # np.savetxt('em_mat', hmm[1], delimiter=',', fmt='%1.3f')
    # print(results)
    # print[results[1]]
    # matprint(hmm[0])
    # matprint(hmm[1])
