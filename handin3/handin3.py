# http://users-cs.au.dk/cstorm/courses/ML_e18/projects/handin3/ml-handin-3.html

import numpy as np

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
    for i in range(0, len(seq)-1):
        #print(i)
        if seq[i] == from_char and seq[i+1] == to_char:
            #print(i)
            count += 1
    return count


def create_count_matrix(gen_arr, char_arr):
    count_mat = np.zeros(shape=(3, 3))
    for x in range(0,len(gen_arr)):
        for i in char_arr:
            for m in char_arr:
                count_mat[char_arr.index(i),char_arr.index(m)] += count_char(list(gen_arr[x][1].items())[0][1],i,m)


def make_trans(count_mat):
    trans_mat = np.zeros(shape=(3, 3))
    for i in range(0,3):
        for x in range(0,3):
            trans_mat[i,x] = count_mat[i,x]/sum(count_mat[i])
    return trans_mat


def chk_next(ann, i):

    if i == len(ann)-1:
        return False
    if ann[i] == ann[i+1]:
        return True
    return False


def extract_seq(seq,ann):
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


def test(seq, ann):
    print('Starting Test with: ')
    print(seq)
    print(ann)
    print('Extracting sequences based on annotation: ')
    result = extract_seq(seq, ann)
    print('N Sequences: ')
    print(result[2])
    print('C Sequences: ')
    print(result[0])
    print('R Sequences: ')
    print(result[1])


    print('Test succes')

if __name__ == '__main__':
    # TODO: make short seq and ann to test input, output of func
    test(seq='CTGACGTCAGCTACGTACGACATGCGTATRC',
         ann='NNNNNNNNNNNCCCCCCCCNNNRRRRRRCNR')

    gen1 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/genome1.fa')
    ann1 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/true-ann1.fa')

    gen2 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/genome2.fa')
    ann2 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/true-ann2.fa')

    gen3 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/genome3.fa')
    ann3 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/true-ann3.fa')

    gen4 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/genome4.fa')
    ann4 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/true-ann4.fa')

    gen5 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/genome5.fa')
    ann5 = read_fasta_file('C:/Users/Ky/Desktop/ml18/handin3/data-handin3/true-ann5.fa')



    gen_arr = [[gen1,ann1],[gen2,ann2],[gen3,ann3],[gen4,ann4],[gen5,ann5]]
    char_arr = ['C','R','N']
    #print(list(gen_arr[0][1].items())[0][1][0])

    #print(list(gen_arr[0][1].items())[0][1][1:500])
    #print(list(gen_arr[0][0].items())[0][1][1:500])
    #result = extract_seq(list(gen_arr[0][0].items())[0][1],list(gen_arr[0][1].items())[0][1])
    #print(result[2])
    #print(result[0])

