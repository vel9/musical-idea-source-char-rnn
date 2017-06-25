import glob
import os.path
import config

from music21 import converter
from shutil import copyfile
from sample import sample_checkpoint
from music21.humdrum.spineParser import SpineLine
from kernhelper import determine_num_voices, should_count_spines, contains_illegal_operations

GENERETED_FILES_DIR = "generated-music"
GENERATED_FILE_PREFIX = "gen"
CHECKPOINTS_DIR = "checkpoints"

def preprocess(kern_content, max_voices):
    """
    Translates the @ symbol back into the standard measure number,
    ensures the the current line does not contain operations
    which would throw off the parsers downstream, like Interpretation lines, 

	see here for more detail on Interpretation lines and 
    the organization of a **kern file in general: 
    http://www.humdrum.org/guide/ch05/
    """
    bar = 1
    line_num = 0
    updated_content = ""
    for line in kern_content.splitlines():
        line = line.strip()
        if (line.startswith(config.MEASURE_SYMBOL)):
            bar_content = copy_append("={}\t".format(bar), max_voices) + "\n"
            updated_content += bar_content
            bar += 1
        elif (line != "" and not contains_illegal_operations(line)):
            spine_line = SpineLine(line_num, line)
            for i in range(spine_line.numSpines, max_voices):
                line += "\t."
            updated_content += line + "\n"
        line_num += 1
    return updated_content

def copy_append(value_to_append, num_copies):
    """
    Copies and appends a value to itself num_copies times
    """
    value = value_to_append
    for i in range(1, num_copies):
        value += value_to_append
    return value

def copy_append_staff(num_copies):
    """
    Copies the staff keyword num_copies amount of times
    Also adds the the staff number number for each keyword
    """
    value = ""
    for i in range(0, num_copies):
        value += "*staff{}\t".format(str(i + 1))
    return value

def generate_krn(kern_content, output_file_name):
    """
    Generates the metadata needed to successfully parse the 
    **krn file, see the following resources: 

    http://www.humdrum.org/guide/ch05/
    http://www.wise.io/tech/asking-rnn-and-ltsm-what-would-mozart-write
    """
    r = []
    num_voices = determine_num_voices(kern_content.splitlines())
    kern_content = preprocess(kern_content, num_voices)
    print("number of voices in generated file: {}".format(num_voices))
    r.append("!!!COM: LIMuse\n")
    r.append("!!!OTL: Fragment\n")
    r.append(copy_append("**kern\t", num_voices) + "\n")
    r.append(copy_append_staff(num_voices) + "\n")
    r.append(copy_append("*Ipiano\t", num_voices) + "\n")
    r.append(copy_append("*clefF4\t", 1) + copy_append("*clefG2\t", num_voices - 1) + "\n")
    r.append(copy_append("*k[]\t\t", num_voices) + "\n")
    r.append(copy_append("*M4/8\t\t", num_voices) + "\n")
    r.append(copy_append("*MM80\t\t", num_voices) + "\n")
    for line in kern_content:
        r.append(line)
    r.append(copy_append("== ", num_voices) + "\n")
    r.append(copy_append("*- ", num_voices) + "\n")
    open(output_file_name,"w").writelines(r)
    print("finished writing krn file")

def generate_midi(music_score, output_midi_filename):
    """
    Converts MusicScore object to midi file
    """
    midi_file = music_score.write("midi")
    copyfile(midi_file, output_midi_filename)
    print("finished writing midi file", midi_file)

def generate_xml(music_score, output_xml_filename):
    """
    Converts MusicScore object to xml file
    """
    mxl_file = music_score.write("musicxml")
    copyfile(mxl_file, output_xml_filename)
    print("finished writing musicxml file", mxl_file)

def generate_files(raw_krn_content, output_krn_filename, output_midi_filename, output_xml_filename):
    """
    Generates the parsable krn file from RNN's raw output

    Once the **kern file is parsed, it is coverted to midi and xml formats
    """
    generate_krn(raw_krn_content, output_krn_filename)
    m = converter.parse(output_krn_filename)
    generate_midi(m, output_midi_filename)
    generate_xml(m, output_xml_filename)

def get_next_file_num(output_krn_dir):
    """
    Scans each file in the output directory and returns the next in sequence
    If gen36.krn is the last file to be generated in the directory, 
    the following will return 37
    """
    max_file_num = 1
    for file_name in os.listdir(output_krn_dir):
        period_idx = file_name.index(".")
        file_num = file_name[len(GENERATED_FILE_PREFIX):period_idx]
        if (file_num.isdigit()):
            max_file_num = max(max_file_num, int(file_num))

    return str(max_file_num + 1)

def ensure_output_directory_exists(directory):
    """
    Creates output directory if it does not exist
    ref: https://stackoverflow.com/a/273227
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate(model_version, checkpoint_name, num_chars, lstm_size, prime=config.MEASURE_SYMBOL):
    """
    Sample the RNN by providing a path to an existing checkpoint
    Use the RNN output to generate the midi and xml files from it 
    """
    generated_dir = GENERETED_FILES_DIR + "/" + model_version
    output_krn_dir = generated_dir + "/krn"
    ensure_output_directory_exists(output_krn_dir)

    output_midi_dir = generated_dir + "/midi"
    ensure_output_directory_exists(output_midi_dir)

    output_xml_dir = generated_dir + "/xml"
    ensure_output_directory_exists(output_xml_dir)
	
	# count the number of files in director and add one 
    output_base_filename = GENERATED_FILE_PREFIX + get_next_file_num(output_krn_dir)

    output_krn_filename = output_krn_dir + "/" + output_base_filename + ".krn"
    output_midi_filename = output_midi_dir + "/" + output_base_filename + ".mid"
    output_xml_filename = output_xml_dir + "/" + output_base_filename + ".xml"
    checkpoint_dir = CHECKPOINTS_DIR + "/" + model_version

    print("sampling from model, and generating new music, base file name: {}".format(output_base_filename))
    raw_krn_content = sample_checkpoint(checkpoint_dir, checkpoint_name, num_chars, lstm_size, prime)
    generate_files(raw_krn_content, output_krn_filename, output_midi_filename, output_xml_filename)
	
if __name__ == "__main__":
    checkpoint_name = "/" + config.CHECKPOINT_NAME
    generate(config.MODEL_VERSION, checkpoint_name, 
        config.OUTPUT_LENGTH, config.LSTM_SIZE, config.PRIME_EVENT)
