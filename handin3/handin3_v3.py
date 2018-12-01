import numpy as np
import math


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


def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)


def translateHid_toindex(path):
    map = {"R": "2", "C": "1", "N": "0"}
    return ''.join(map[idx] for idx in path)


def map_indexes_to_states(obs):
    mapping = {0: 'N', 1: 'C', 2: 'C', 3: 'C', 4: 'C', 5: 'C',
               6: 'C', 7: 'C', 8: 'C', 9: 'C', 10: 'C', 11: 'C',
               12: 'C', 13: 'C', 14: 'C', 15: 'C', 16: 'C',
               17: 'C', 18: 'C', 19: 'C', 20: 'C', 21: 'C',
               22: 'R', 23: 'R', 24: 'R', 25: 'R', 26: 'R', 27: 'R',
               28: 'R', 29: 'R', 30: 'R', 31: 'R', 32: 'R', 33: 'R',
               34: 'R', 35: 'R', 36: 'R', 37: 'R', 38: 'R',
               39: 'R', 40: 'R', 41: 'R', 42: 'R'}
    return ''.join(mapping[idx] for idx in obs)


# Transform Genome sequence to indexes
def translateObs_toindex(obs):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [mapping[symbol.upper()] for symbol in obs]


# Return integer corresponding to base
def getNucleotide(integer_index):
    mapping = ['A', 'C', 'G', 'T']
    return mapping[integer_index]


# return stop or start codon for coding sequence based on 3 nucleotides in form of indexes
def find_stop_or_start_coding_codon(first, second, third, start=True):
    dict_start = {"ATG": 1, "GTG": 16, "TTG": 19}
    dict_stop = {"TAG": 7, "TAA": 10, "TGA": 13}
    if start:
        for stop in dict_start:
            if (getNucleotide(first) + getNucleotide(second) + getNucleotide(third)) == stop:
                return dict_start[stop]
    else:
        for stop in dict_stop:
            if (getNucleotide(first) + getNucleotide(second) + getNucleotide(third)) == stop:
                return dict_stop[stop]


# return stop or start codon for reverse sequence based on 3 nucleotides in form of indexes
def find_stop_or_start_reverse_codon(first, second, third, start=True):
    dict_stop = {"TTA": 28, "CTA": 31, "TCA": 34}
    dict_start = {"CAT": 22, "CAC": 37, "CAA": 40}
    if start:
        for stop in dict_start:
            if (getNucleotide(first) + getNucleotide(second) + getNucleotide(third)) == stop:
                return dict_start[stop]
    else:
        for stop in dict_stop:
            if (getNucleotide(first) + getNucleotide(second) + getNucleotide(third)) == stop:
                return dict_stop[stop]


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


def starting_model():
    K = 43
    P = 4
    init_probs = np.ones((K))
    trans_probs = np.ones((K, K))
    emission_probs = np.ones((K, P))

    # Fix probabilities of transitions
    for i in range(16):
        if i != 0 and i != 6 and i != 9 and i != 12 and i != 15:
            trans_probs[i][i + 1] = 1

    return init_probs, trans_probs, emission_probs


def determine_first_state(obs, hid, init, emis):
    if hid[0] == "0":
        init[0] += 1  # Add 1 to iniciate with non-coding
        emis[0][obs[0]] += 1  # Add 1 to emiting the base from the non-coding region
    return init, emis


def update_coding_params(hid, obs, n, trans, emis):
    if hid[n - 1] == "0":  # Start codon
        start = find_stop_or_start_coding_codon(obs[n], obs[n + 1], obs[n + 2], start=True)
        if start:
            trans[0][start] += 1
        n += 3

    elif hid[n + 3] == "0":  # Stop codon
        stop = find_stop_or_start_coding_codon(obs[n], obs[n + 1], obs[n + 2], start=False)
        if stop:
            trans[6][stop] += 1
        n += 3
    else:
        trans[6][4] += 1
        emis[4][obs[n]] += 1
        emis[5][obs[n + 1]] += 1
        emis[6][obs[n + 2]] += 1
        n += 3

    return n, trans, emis


def update_reverse_params(hid, obs, n, trans, emis):
    if hid[n - 1] == "0":  # Start codon
        stop = find_stop_or_start_reverse_codon(obs[n], obs[n + 1], obs[n + 2], start=True)
        if stop:
            trans[0][stop] += 1
        n += 3

    elif hid[n + 3] == "0":  # Stop codon
        start = find_stop_or_start_reverse_codon(obs[n], obs[n + 1], obs[n + 2], start=False)
        if start:
            trans[27][start] += 1
        n += 3
    else:
        trans[27][25] += 1
        emis[25][obs[n]] += 1
        emis[26][obs[n + 1]] += 1
        emis[27][obs[n + 2]] += 1
        n += 3

    return n, trans, emis


def update_model_by_counting(genome, ann, init, trans, emis):
    """
    0: Non-coding

    1: START (1) - A
    2: START (1) - T
    3: START (1) - G
    4: CODING - position 1
    5: CODING - position 2
    6: CODING - position 3
    7: STOP (1) - T
    8: STOP (1) - A
    9: STOP (1) - A
    10: STOP (2) - T
    11: STOP (2) - A
    12: STOP (2) - G
    13: STOP (3) - T
    14: STOP (3) - G
    15: STOP (3) - A
    16: START (2) - G
    17: START (2) - T
    18: START (2) - G
    19: START (3) - T
    20: START (3) - T
    21: START (3) - G


    22: START-R (1) - C
    23: START-R (1) - A
    24: START-R (1) - T
    25: REVERSE - position 1
    26: REVERSE - position 2
    27: REVERSE - position 3
    28: STOP-R (1) - T
    29: STOP-R (1) - T
    30: STOP-R (1) - A
    31: STOP-R (2) - C
    32: STOP-R (2) - T
    33: STOP-R (2) - A
    34: STOP-R (3) - T
    35: STOP-R (3) - C
    36: STOP-R (3) - A

    37: START-R (2) - C
    38: START-R (2) - A
    39: START-R (2) - C

    40: START-R (3) - C
    41: START-R (3) - A
    42: START-R (3) - A

    """

    obs = translateObs_toindex(genome)
    hidden = translateHid_toindex(ann)
    init, emis = determine_first_state(obs, hidden, init, emis)
    # start from 1 since we determined first state
    n = 1
    while n < len(obs) - 3:  # go until last codon, with 3 possible cases (N,C,R)
        if hidden[n] == "0":  # Stay in non-coding
            trans[0][0] += 1  # count staying in non coding, inc by 1
            emis[0][obs[n]] += 1  # add emission in non coding state
            n += 1
        elif hidden[n] == "1":  # Coding seq
            n, trans, emis = update_coding_params(hidden, obs, n, trans, emis)
        elif hidden[n] == "2":  # Reverse
            n, trans, emis = update_reverse_params(hidden, obs, n, trans, emis)

    return init, trans, emis


def viterbi(init_probs, trans_probs, emission_probs, seq):
    k = len(init_probs)
    n = len(seq)

    w = [[0] * n for _ in range(k)]

    for i in range(k):
        w[i][0] = log(init_probs[i]) + log(emission_probs[i][seq[0]])

    for j in range(1, n):
        for i in range(k):
            w[i][j] = max([log(emission_probs[i][seq[j]]) + w[g][j - 1] + log(trans_probs[g][i]) for g in range(k)])

    N = len(w[0])
    K = len(w)

    #p = max(w[i][-1] for i in range(len(w)))
    backtrack = [0 for k in range(N)]

    backtrack[N - 1] = max(range(K), key=lambda k: w[k][N - 1])

    for i in range(N - 2, -1, -1):
        backtrack[i] = max(range(K), key=lambda k:
        log(emission_probs[backtrack[i + 1]][seq[i + 1]]) + w[k][i] + log(trans_probs[k][backtrack[i + 1]]))
        print(backtrack[i])

    return backtrack


def updateMatrices(old_init, init, old_trans, trans, old_emis, emis):
    for i in range(0, len(old_trans)):
        for j in range(0, len(old_trans)):
            old_trans[i][j] = old_trans[i][j] + trans[i][j]

    for i in range(0, len(old_emis)):
        for j in range(0, len(old_emis[1])):
            old_emis[i][j] = old_emis[i][j] + emis[i][j]

    for i in range(0, len(old_init)):
        old_init[i] = old_init[i] + init[i]

    return old_init, old_trans, old_emis


def cross_validate(list_of_genAnn, only_train=True):
    init, trans, emis = starting_model()  # starting model assumes that we saw everything at least once, so we ignore 0 probs
    if only_train:
        for i in list_of_genAnn:
            counts = update_model_by_counting(i[0], i[1], init, trans, emis)
            init, trans, emis = updateMatrices(init, counts[0], trans, counts[1], emis, counts[2])

        trans_probs = np.divide(trans, trans.sum(axis=1, keepdims=True),
                                out=np.zeros_like(trans),
                                where=trans.sum(axis=1, keepdims=True) != 0)
        emission_probs = np.divide(emis, emis.sum(axis=1, keepdims=True),
                                   out=np.zeros_like(emis),
                                   where=emis.sum(axis=1, keepdims=True) != 0)
        init_probs = np.array(init / init.sum())

    else:
        counter = 1
        for i in list_of_genAnn:
            for j in list_of_genAnn:
                if i != j:  # not diagonal
                    print('train on (counting...): ', j[0][0:10], "...", j[1][0:10], "....")
                    counts = update_model_by_counting(j[0], j[1], init, trans, emis)
                    init, trans, emis = updateMatrices(init, counts[0], trans, counts[1], emis, counts[2])

            trans_probs = np.divide(trans, trans.sum(axis=1, keepdims=True),
                                    out=np.zeros_like(trans),
                                    where=trans.sum(axis=1, keepdims=True) != 0)
            emission_probs = np.divide(emis, emis.sum(axis=1, keepdims=True),
                                       out=np.zeros_like(emis),
                                       where=emis.sum(axis=1, keepdims=True) != 0)
            init_probs = np.array(init / init.sum())

            print('validate on:', i[1][0:10], "...")
            obs = translateObs_toindex(i[0])
            viterbi_result = viterbi(init_probs, trans_probs, emission_probs, obs)
            viterbi_seq = map_indexes_to_states(viterbi_result)
            print_all(i[1], viterbi_seq)

            #decoding_gen = open("decoding_gen" + str(counter) + '.fa', 'x')
            #decoding_gen.write("> pred-ann" + str(counter) + "\n" + viterbi_seq)
            counter += 1

    return init_probs, trans_probs, emission_probs


def predict(init_probs, trans_probs, emission_probs, unknown):
    counter = 6
    for genome in unknown:
        obs = translateObs_toindex(genome)
        viterbi_result = viterbi(init_probs, trans_probs, emission_probs, obs)
        viterbi_seq = map_indexes_to_states(viterbi_result)
        decoding_gen = open("decoding_gen" + str(counter) + '.fa', 'x')
        decoding_gen.write("> pred-ann" + str(counter) + "\n" + viterbi_seq)
        counter += 1


if __name__ == '__main__':
    char_arr = ['C', 'R', 'N']
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

    init_probs, trans_probs, emiss_probs = cross_validate(gen_arr, only_train=False)

    obs = translateObs_toindex(gen1['genome1'])

    viterbi_result = viterbi(init_probs, trans_probs.T, emiss_probs, obs)
    viterbi_seq = map_indexes_to_states(viterbi_result)
    print(viterbi_seq)
    print_all(ann1['true-ann1'], viterbi_seq)

    # predict(init_probs, trans_probs, emiss_probs, unknown_arr)
