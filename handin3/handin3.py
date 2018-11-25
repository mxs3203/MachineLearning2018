# http://users-cs.au.dk/cstorm/courses/ML_e18/projects/handin3/ml-handin-3.html
import math

import numpy as np
from collections import Counter

TESTING = False


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


def count_char(seq, from_char, to_char):
    count = 0
    for i in range(0, len(seq) - 1):
        if seq[i] == from_char and seq[i + 1] == to_char:
            count += 1
    return count


def create_count_matrix(seq, ann):
    char_arr = ['C', 'R', 'N']
    count_mat = np.zeros(shape=(3, 3))
    for i in range(3):
        for n in range(3):
            count_mat += count_char(seq,char_arr[i],char_arr[n])

    return (count_mat)


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
    c_list, r_list, n_list = extract_seq(seq, ann)
    c_start_codons = []
    c_stop_codons = []
    r_start_codons = []
    r_stop_codons = []
    c_new = []
    r_new= []
    c_counts = [0, 0, 0, 0] #ATCG
    r_counts = [0, 0, 0, 0]
    n_counts = [0, 0, 0, 0]

    for x in range(0, len(c_list)):
        i = c_list[x]
        c_start_codons.append(i[0:3])
        c_stop_codons.append(i[len(i)-3, len(i)])
        c_new.append(i[3:-3])

    for x in range(0, len(r_list)):
        i = r_list[x]
        r_start_codons.append(i[0:3])
        r_stop_codons.append(i[len(i)-3, len(i)])
        r_list.new(i[3:-3])

    for i in c_new:
        c_counts[0] += i.count('A')
        c_counts[1] += i.count('T')
        c_counts[2] += i.count('C')
        c_counts[3] += i.count('G')

    for i in r_new:
        r_counts[0] += i.count('A')
        r_counts[1] += i.count('T')
        r_counts[2] += i.count('C')
        r_counts[3] += i.count('G')

    for i in n_list:
        n_counts[0] += i.count('A')
        n_counts[1] += i.count('T')
        n_counts[2] += i.count('C')
        n_counts[3] += i.count('G')

    for i in range(0,len(c_counts)):
        length = sum(c_counts)
        c_counts[i] = c_counts[i]/length

    for i in range(0,len(r_counts)):
        length = sum(r_counts)
        r_counts[i] = r_counts[i]/length

    for i in range(0,len(n_counts)):
        length = sum(n_counts)
        n_counts[i] = n_counts[i]/length

    return Counter(c_start_codons), Counter(r_start_codons), c_counts, r_counts, n_counts

def make_hmm(old_trans, old_emat):

    new_trans = np.zeros(shape=(15,15))
    for i in range(14):
        new_trans[i,i+1] = 1
    new_trans[0,0] = old_trans[2,2]
    new_trans[7,0] = old_trans[0,2]
    new_trans[14,0] = old_trans[1,2]
    new_trans[14,1] = old_trans[1,0]
    new_trans[0,1] = old_trans[2,0]
    new_trans[4,4] = old_trans[0,0]
    new_trans[4,5] = old_trans[0,1] + old_trans[0,2]
    new_trans[0,8] = old_trans[2,1]
    new_trans[7,8] = old_trans[0,1]
    new_trans[11,11] = old_trans[1,1]
    new_trans[11,12] = old_trans[1,0] + old_trans[1,2]

    new_emat = np.zeros(shape=(4,15))
    new_emat[0,] = [0,1]+[0]*4+[1,1]+[0]*2+[1]+[0]*2+[1,0]
    new_emat[1,] = [0]*2+[1]+[0]*2+[1]+[0]*2+[1,1]+[0]*4+[1]
    new_emat[2,] = [0]*12+[1]+[0,0]
    new_emat[3,] = [0]*3+[1]+[0]*11
    new_emat[:,0] = old_emat[:,0]
    new_emat[:,4] = old_emat[:,1]
    new_emat[:,11] = old_emat[:,2]

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


def viterbi(trans_matrix, emission_matrix, init_prob, seq):

    seq = translate_observations_to_indices(seq)
    seqLen = len(seq)
    nHiddenStates = trans_matrix.shape[0]


    viterbi_seq = [-1] * seqLen
    score_matrix = np.zeros(shape=(seqLen, nHiddenStates))

    score_matrix[0,: ] = init_prob * emission_matrix[seq[0], :]

    for charInSeq in range(1,seqLen):
        for hidState in range(0,nHiddenStates):
            previousRow = score_matrix[charInSeq-1, :]
            emission4givenState = emission_matrix[seq[charInSeq], hidState]
            # trans probs of coming to specified hidState
            transitionProb = trans_matrix[:, hidState]
            temp = [emission4givenState] * nHiddenStates
            product = previousRow * transitionProb * temp
            score_matrix[charInSeq, hidState] = max(product)

    return np.argmax(score_matrix, axis=1)

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
    #assert result[0][0] == expected[0][0], result[0][1] == expected[0][1]
    #assert result[0][2] == expected[0][2], result[1][0] == expected[1][0]
    #assert result[2][0] == expected[2][0], result[2][1] == expected[2][1]
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
    #assert result[0][0] == expected[0][0], result[0][1] == expected[0][1]
    #assert result[0][2] == expected[0][2], result[1][0] == expected[1][0]
    #assert result[2][0] == expected[2][0], result[2][1] == expected[2][1]
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
    if TESTING:
        test_exstract_seq(seq='CTGACGTCAGCTACGTACGACATGCGTATTC',
                          ann='NNNNNNNNNNNCCCCCCCCNNNRRRRRRRRN')

        # states of length 10
        gen_arr = [[{'genome1': "GTCAGTACGT"}, {'true-ann1': "NNNNNNNNNN"}],  # C->R = 1, N->C = 1, R->N=1 Others 0
                   [{'genome2': "AGTAAAACGT"}, {'true-ann2': "RRRRRRRRRR"}],  # 11 R to R
                   [{'genome3': "TGTAAAACGT"}, {'true-ann3': "CCCCCCCCCC"}],  # 11 C to C
                   [{'genome4': "TGTAAAACGT"}, {'true-ann4': "NNNCCCRRRN"}]]  # 11 N to N

        test_count_char("NNNNCCCCNCRRRNCRR", 'N', 'C', 3)
        test_count_char("NNNNCCCCNCRRRCRCR", 'R', 'C', 2)
        count_mat = test_count_mat(gen_arr, char_arr)
        test_make_trans(count_mat)
        test_emis_matrix(seq='CTGACGTCAGCTACGTACGACATGCGTATTC',
                         ann='NNNNNNNNNNNCCCCCCCCNNNRRRRRRCNR')

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        emis_cnt = make_emission_count(seq='CTGACGTCAGCTACGTACGACATGCGTATTC',
                                       ann='NNNNNNNNNNNCCCCCCCCNNNRRRRRRRRR')
        print(emis_cnt)
        emis_prob = make_emission_prob_matrix(emis_cnt)
        sample = [[{'genome1': "CTGACGTCAGCTACGTACGACATGCGTATTC"}, {'true-ann1': "NNNNNNNNNNNCCCCCCCCNNNRRRRRRRRR"}]]
        trans_cnt = create_count_matrix(sample, char_arr)
        trans_matrix = make_trans(trans_cnt)

        viterbi_seq = viterbi(emission_matrix=emis_prob,
                              init_prob=[1 / 3, 1 / 3, 1 / 3],
                              trans_matrix=trans_matrix,
                              seq='AGTCAGCTGCTA')
        print(viterbi_seq)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    ####################################################################
    else: # NOT TESTING
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

        gen_arr = [[gen1, ann1], [gen2, ann2], [gen3, ann3], [gen4, ann4], [gen5, ann5]]
        char_arr = ['C', 'R', 'N']
        # print(list(gen_arr[0][0].items())[0][0])

        # c_list, r_list, n_list, start_codons_c, end_codons_c, start_codons_r, end_codons_r
        c_list, r_list, n_list, c_start, c_end, r_start, r_end = [], [], [], [], [], [], []
        trans_counts = np.zeros(shape=(3, 3))
        emission_counts = np.zeros(shape=(3, 4))

        for i in range(5):
            seq, ann = list(gen_arr[i][0].items())[0][1], list(gen_arr[i][1].items())[0][1]
            result = extract_seq(seq, ann)
            c_list.append(result[0])
            r_list.append(result[1])
            n_list.append(result[2])
            c_start.append(result[3])
            c_end.append(result[4])
            r_start.append(result[5])
            r_end.append(result[6])

            trans_counts += create_count_matrix(seq, ann)
            emission_counts += make_emission_count(seq, ann)

        trans_mat = make_trans(trans_counts)
        em_mat = make_emission_prob_matrix(emission_counts)

        hmm = make_hmm(trans_mat, em_mat.transpose())
        viterbi_seq = viterbi(hmm[0], hmm[1], init_prob=[1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15], seq='CAGTACGTACGTACGATCGATCGAT')
        print(viterbi_seq)