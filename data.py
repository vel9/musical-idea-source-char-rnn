import glob
import numpy as np
import _pickle as pickle

from pathlib import Path

VOCAB_FILE_NAME = "vocab.p"
VOCAB_TO_INT_FILE_NAME = "vocab_to_int.p"
INT_TO_VOCAB_FILE_NAME = "int_to_vocab.p"
ENCODED_VOCAB_FILE_NAME = "encoded.p"
DATASET_DIR = "dataset"
DATASET_FILENAME = "music.txt"
# Provide list of composers whose music we'll be processing into the training data
COMPOSERS = ["beethoven", "chopin", "haydn", "mozart", "prokofiev",
"ravel", "scarlatti","schubert","scriabin"]

def get_output_dir(args):
    """
    Provides the output directory for files generated from 
    the data processing phase (pickle outputs of the processed/encoded dataset)
    as well as the the training phase (the tensorflow checkpoints)

    The "local" output is placed in the relative output directory 
    Floydhub, on the other hand requires outputs to be placed in the "/output"
    directory, as this is the "only path Floyd will preserver"

    See more: http://docs.floydhub.com/commands/output/ 
    """
    output_dir_dict = {"local": "output", "floyd": "/output"}
    if len(args) < 2:
        raise ValueError("please provide an env argument [local or floyd]")
    else:
        env_args = (args)[1]
        return output_dir_dict[env_args]

def read_file(file_path):
    """
    Loads pickle file from file_path, returns None if it doesn't exist
    """
    try:
        file_content =  pickle.load(open(file_path, "rb"))
        print("loading {} from file system".format(file_path))
        return file_content
    except (OSError, IOError) as e:
        print("No file found at {}".format(file_path))
        return None

def write_file(file_path, file_content):
    """
    Writes file content to file_path
    """
    pickle.dump(file_content, open(file_path, "wb"))
    print("saving new file: {} to file system".format(file_path))

def read_processed_data(directory):
    """
    Reads already-processed files from file system
    """
    directory = directory + "/"
    vocab = read_file(directory + VOCAB_FILE_NAME)
    vocab_to_int = read_file(directory + VOCAB_TO_INT_FILE_NAME)
    int_to_vocab = read_file(directory + INT_TO_VOCAB_FILE_NAME)
    encoded = read_file(directory + ENCODED_VOCAB_FILE_NAME)
    return vocab, vocab_to_int, int_to_vocab, encoded

def generate_vocab(output_dir, text):
    """
    Generates a unique set of all characters found in the dataset text
    """
    vocab_file_path = output_dir + VOCAB_FILE_NAME
    vocab = read_file(vocab_file_path)
    if (vocab is None):
        vocab = set(text)
        write_file(vocab_file_path, vocab)
    return vocab

def generate_vocab_to_int(output_dir, vocab):
    """
    Maps each character to an integer value: {"b":1}
    """
    vocab_to_int_file_path = output_dir + VOCAB_TO_INT_FILE_NAME
    vocab_to_int = read_file(vocab_to_int_file_path)
    if (vocab_to_int is None):
        vocab_to_int = {c: i for i, c in enumerate(vocab)}
        write_file(vocab_to_int_file_path, vocab_to_int)
    return vocab_to_int

def generate_int_to_vocab(output_dir, vocab_to_int):
    """
    Crates a reverse mapping from vocab_to_int
    ref: https://stackoverflow.com/a/2569074
    """
    int_to_vocab_file_path = output_dir + INT_TO_VOCAB_FILE_NAME
    int_to_vocab = read_file(int_to_vocab_file_path)
    if (int_to_vocab is None):
        int_to_vocab = {v: k for k, v in vocab_to_int.items()}
        write_file(int_to_vocab_file_path, int_to_vocab)
    return int_to_vocab

def generate_encoded(output_dir, vocab_to_int, text):
    """
    Encodes the input dataset text into integers as mapped in vocab_to_int
    Sequence "bbc" would become [1,1,2]
    """
    encoded_file_path = output_dir + ENCODED_VOCAB_FILE_NAME
    encoded = read_file(encoded_file_path)
    if (encoded is None):
        encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
        write_file(encoded_file_path, encoded)
    return encoded

def process(text, output_dir):
    """
    Processes input data into data structures needed for the training process
    """
    output_dir = output_dir + "/"
    vocab = generate_vocab(output_dir, text)
    vocab_to_int = generate_vocab_to_int(output_dir, vocab)
    int_to_vocab = generate_int_to_vocab(output_dir, vocab_to_int)
    encoded = generate_encoded(output_dir, vocab_to_int, text)
    return vocab, vocab_to_int, int_to_vocab, encoded

def generate_data(output_dir):
    """
    Processing code primarily adopted from: 
    http://www.wise.io/tech/asking-rnn-and-ltsm-what-would-mozart-write

    Reads **kern files from a local dataset, the files are placed within 
    folders named by composer names: dataset/mosart/[mozart's compositions]
    
    Crucially, the process replaces measure numbers in the **kern file with the
    @ symbol, hiding the measure numbers itself from the training process. 

    If the generated dataset already exists, use it
    """
    dataset_dir = DATASET_DIR
    dataset_file_name = DATASET_FILENAME
    music_txt = Path(dataset_file_name)
    if music_txt.is_file():
        with open(dataset_file_name, "r") as existing_data_file:
            existing_data = existing_data_file.read()
            if (len(existing_data) > 0):
                print("found existing dataset...")
                return process(existing_data, output_dir)

    print("generating dataset...")
    comp_txt = open(dataset_file_name, "w")
    # ref: http://www.wise.io/tech/asking-rnn-and-ltsm-what-would-mozart-write
    # get the top directory ites
    for composer in COMPOSERS:
        compositions = glob.glob(dataset_dir + "/{composer}/*.krn".format(composer=composer))
        for composition in compositions:
            lines = open(composition, "r").readlines()
            out = []
            found_first_measure = False
            for l in lines:
                if l.startswith("="):
                    out.append(config.MEASURE_SYMBOL + "\n")
                    found_first_measure = True
                    continue
                if not found_first_measure or l.startswith("!"):
                    ## skip line until we find the end of the header and metadata
                    ## and ignore comments
                    continue
                out.append(l)
            comp_txt.writelines(out)
    comp_txt.close()

    with open(dataset_file_name, "r") as f:
        text = f.read()
    return process(text, output_dir)
