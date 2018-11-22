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


#gen_dict =  {'genome1': [gen1,ann1],'genome2': [gen2,ann2],'genome3': [gen3,ann3],'genome4': [gen4,ann4],'genome5': [gen5,ann5]}
gen_arr = [[gen1,ann1],[gen2,ann2],[gen3,ann3],[gen4,ann4],[gen5,ann5],]
#print(gen_dict['genome1'][0])

count_mat = np.zeros(shape=(3,3))
print(type(ann1))
print(list(gen_arr[0][1].items())[0][1][0])
#for i in gen_dict:

char_arr = ['C','R','N']

def count_char(seq,from_char,to_char):
    count = 0
    #print(seq[1:5])
    for i in range(0, len(seq)-1):
        #print(i)
        if seq[i] == from_char and seq[i+1] == to_char:
            #print(i)
            count += 1
    return count

#print(count_char(list(ann1.items())[0][1],'R','R'))




for x in range(0,len(gen_arr)):
    for i in char_arr:
        for m in char_arr:
            count_mat[char_arr.index(i),char_arr.index(m)] += count_char(list(gen_arr[x][1].items())[0][1],i,m)

print(count_mat)




