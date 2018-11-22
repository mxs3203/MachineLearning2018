# http://users-cs.au.dk/cstorm/courses/ML_e18/projects/handin3/ml-handin-3.html

import numpy as np

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


def create_count_matrix(gen_arr, char_arr):
    count_mat = np.zeros(shape=(3, 3))
    for x in range(0, len(gen_arr)):
        for i in char_arr:
            for m in char_arr:
                count_mat[char_arr.index(i), char_arr.index(m)] += count_char(list(gen_arr[x][1].items())[0][1], i, m)

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


def extract_seq(seq, ann):
    i = 0
    start = 0
    end = 0
    n_list = []
    r_list = []
    c_list = []
    while i != len(ann):
        if ann[i] == 'N' and not chk_next(ann, i):
            end = i + 1
            n_list.append(seq[start:end])
            start = i + 1
        if ann[i] == 'C' and not chk_next(ann, i):
            end = i + 1
            c_list.append(seq[start:end])
            start = i + 1
        if ann[i] == 'R' and not chk_next(ann, i):
            end = i + 1
            r_list.append(seq[start:end])
            start = i + 1
        i += 1

    return c_list, r_list, n_list


def test_exstract_seq(seq, ann):
    print('-------------Starting Test  extract Seq with:------------- ')
    print(seq)
    print(ann)
    print('Extracting sequences based on annotation: ')
    result = extract_seq(seq, ann)

    assert result[0] == ['TACGTACG', 'T']
    assert result[1] == ['TGCGTA', 'C']
    assert result[2] == ['CTGACGTCAGC', 'ACA', 'T']

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
    assert result[0][0] == expected[0][0], result[0][1] == expected[0][1]
    assert result[0][2] == expected[0][2], result[1][0] == expected[1][0]
    assert result[2][0] == expected[2][0], result[2][1] == expected[2][1]
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
    assert result[0][0] == expected[0][0], result[0][1] == expected[0][1]
    assert result[0][2] == expected[0][2], result[1][0] == expected[1][0]
    assert result[2][0] == expected[2][0], result[2][1] == expected[2][1]
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


if __name__ == '__main__':

    ################# Unit Testing ####################################
    if TESTING:
        test_exstract_seq(seq='CTGACGTCAGCTACGTACGACATGCGTATTC',
                          ann='NNNNNNNNNNNCCCCCCCCNNNRRRRRRCNR')

        # states of length 10
        gen_arr = [[{'genome1': "GTCAGTACGT"}, {'true-ann1': "NNNNNNNNNN"}],  # C->R = 1, N->C = 1, R->N=1 Others 0
                   [{'genome2': "AGTAAAACGT"}, {'true-ann2': "RRRRRRRRRR"}],  # 11 R to R
                   [{'genome3': "TGTAAAACGT"}, {'true-ann3': "CCCCCCCCCC"}],  # 11 C to C
                   [{'genome4': "TGTAAAACGT"}, {'true-ann4': "NNNCCCRRRN"}]]  # 11 N to N
        char_arr = ['C', 'R', 'N']
        test_count_char("NNNNCCCCNCRRRNCRR", 'N', 'C', 3)
        test_count_char("NNNNCCCCNCRRRCRCR", 'R', 'C', 2)
        count_mat = test_count_mat(gen_arr, char_arr)
        test_make_trans(count_mat)
    ####################################################################

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

    # TODO: Chnage this data structure if we have time!
    gen_arr = [[gen1, ann1], [gen2, ann2], [gen3, ann3], [gen4, ann4], [gen5, ann5]]
    char_arr = ['C', 'R', 'N']
    # print(list(gen_arr[0][1].items())[0][1][0])

    result = extract_seq(list(gen_arr[0][0].items())[0][1], list(gen_arr[0][1].items())[0][1])
    c_list = result[0]
    r_list = result[1]
    n_list = result[2]
