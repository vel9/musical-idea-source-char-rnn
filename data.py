# This network is from Udacity's Intro to RNN project

# data from https://github.com/automata/ana-music
import glob
import numpy as np
import _pickle as pickle

from pathlib import Path

vocab_file_name = '/vocab.p'
vocab_to_int_file_name = '/vocab_to_int.p'
int_to_vocab_file_name = '/int_to_vocab.p'
encoded_file_name = '/encoded.p'

def get_output_dir(args):
    output_dir_dict = {"local": "output", "floyd": "/output"}
    if len(args) < 2:
        raise ValueError("please provide an env argument [local or floyd]")
    else:
        env_args = (args)[1]
        output_dir = output_dir_dict[env_args]
        return output_dir

def perhaps_read(file_name, file_content):
    try:
        file_content =  pickle.load(open(file_name, "rb"))
        print("loading {} from file-system".format(file_name))
        return file_content
    except (OSError, IOError) as e:
        pickle.dump(file_content, open(file_name, "wb"))
        print("saving new file: {} to file-system".format(file_name))
        return file_content

def read_file(file_name):
    return pickle.load(open(file_name, "rb"))

def read_processed_data(directory):
    vocab = read_file(directory + vocab_file_name)
    vocab_to_int = read_file(directory + vocab_to_int_file_name)
    int_to_vocab = read_file(directory + int_to_vocab_file_name)
    encoded = read_file(directory + encoded_file_name)
    return vocab, vocab_to_int, int_to_vocab, encoded

def process(text, output_dir):
    vocab = perhaps_read(output_dir + vocab_file_name, set(text))
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    vocab_to_int = perhaps_read(output_dir + vocab_to_int_file_name, vocab_to_int)
    int_to_vocab = perhaps_read(output_dir + int_to_vocab_file_name, dict(enumerate(vocab)))
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
    encoded = perhaps_read(output_dir + encoded_file_name, encoded)
    return vocab, vocab_to_int, int_to_vocab, encoded

def generate_data(output_dir):
    dataset = 'dataset2'
    ddataset_file_name = 'music.txt'
    music_txt = Path(ddataset_file_name)
    if music_txt.is_file():
        with open(datase_file_name, 'r') as existing_data_file:
            existing_data = existing_data_file.read()
            if (len(existing_data) > 0):
                print("found existing dataset...")
                return process(existing_data, output_dir)

    print("generating dataset...")
    composers = ["beethoven",
                "chopin",
                "haydn",
                "mozart",
                "prokofiev",
                "ravel",
                "scarlatti",
                "schubert",
                "scriabin"]

    comp_txt = open(ddataset_file_name, "w")
    # ref: http://www.wise.io/tech/asking-rnn-and-ltsm-what-would-mozart-write
    # get the top directory ites
    for composer in composers:
        ll = glob.glob(dataset + '/{composer}/*.krn'.format(composer=composer))
        for song in ll:
            lines = open(song,"r").readlines()
            out = []
            found_first = False
            for l in lines:
                if l.startswith("="):
                    ## new measure, replace the measure with the @ sign
                    out.append("@\n")
                    found_first = True
                    continue
                if not found_first or l.startswith("!"):
                    ## keep going until we find the end of the header and metadata
                    ## and ignore comments
                    continue
                out.append(l)
            comp_txt.writelines(out)
    comp_txt.close()

    with open(datase_file_name, 'r') as f:
        text = f.read()
    return process(text, output_dir)
