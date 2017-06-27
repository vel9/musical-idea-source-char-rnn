# Algorithm for gettting some insight into how much a given 
# generated sequence has in common with the dataset the model
# was trained on.
#
# Designed for use with the model (v25) trained on monophonic data
# 
# The analysis can be executed for a single generated sequence 
# or for directory of generated sequences
#
import os.path
from data import collect_krn_files

DATASET_DIR = "dataset/monophonic"
MIN_SEQ_LENGTH = 10 # characters
MAX_SEQ_LENGTH = 40 # characters
NOTES_IN_SEQ = 5

def read_file(file_path):
    """
    Reads file from file-system, returns None if it encounters
    an exception (such as an encoding problme) 
    """
    try:
        return open(file_path, "r").read()
    except (IOError, UnicodeDecodeError) as e:
        #print("Error reading file: {}, e: {}".format(file_path, e))
        return None

def is_valid_sequence_start(text, i):
    """
    Valid sequences are ones which start with a number
    and the next character is NOT a number
    """
    return i > 0 and (text[i].isdigit() and not text[i - 1].isdigit())

def get_num_notes(note_sequences):
    """
    Returns number of notes in given sequence
    """
    return len(note_sequences.splitlines()) 

def is_valid_sequence_end(text, i, j):
    """
    Valid sequences are ones which start with a number
    and the next character is NOT a number
    """
    if text[j] == "\n":
        substring_to_check = text[i:j]
        return get_num_notes(substring_to_check) >= NOTES_IN_SEQ

    return False

def analyze_common_sequences(krn_file):
    """
    Provides a mini-report for how many long sequences contained within
    the provided krn_file appear elsewhere in the dataset

    Builds on the usual "needle-in-the-haystack" algorithm so it runs slowly
    if your dataset is huge, but I wanted to implemnt this myself
    for some additional fine-grained control
    """
    print("****")
    print("checking {} for common phrases against dataset {}\n".format(krn_file, DATASET_DIR))
    with open(krn_file, "r") as f:
        text = f.read()

    text_len = len(text)
    total_common_sequences = 0
    compositions = collect_krn_files(DATASET_DIR)
    for comp_file in compositions:
        composition = read_file(comp_file)
        if (composition is None):
            continue

        counter = 0
        already_checked = {}
        for i in range(0, text_len - MIN_SEQ_LENGTH):
            if (not is_valid_sequence_start(text, i)):
                continue

            jstart = i + MIN_SEQ_LENGTH
            jend = min(i + MAX_SEQ_LENGTH, text_len)
            for j in reversed(range(jstart, jend)):
                if (not is_valid_sequence_end(text, i, j)):
                    continue

                substring_to_check = text[i:j]
                if substring_to_check in already_checked:
                    print("already checked {}".format(substring_to_check))
                    continue

                if substring_to_check in composition: 
                    already_checked[substring_to_check] = True
                    print("found sequence, len: {}\n{}\n"
                                .format(str(get_num_notes(substring_to_check)), substring_to_check))
                    counter += 1
                    break

        if (counter > 0):    
            print("number of common sequences in {}: {}".format(comp_file, counter))
            print("---")
            total_common_sequences += counter

    print("total common sequences: {}".format(total_common_sequences))
    print("****\n")

def analyze_common_sequences_dir(kern_files_to_check_dir):
    """
    Analyzes common sequences for each ".krn" file in 
    the provided directory
    """
    file_extension = ".krn"
    for file in os.listdir(kern_files_to_check_dir):
        # ignore non-krn files
        if (file[-len(file_extension):] != file_extension):
            continue

        analyze_common_sequences(kern_files_to_check_dir + "/" + file)

if __name__ == "__main__":
    analyze_common_sequences_dir("generated-music/v25/krn")
