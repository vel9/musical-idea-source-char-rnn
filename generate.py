import glob
import os.path

from music21 import converter
from shutil import copyfile
from sample import sample_checkpoint
from music21.humdrum.spineParser import SpineLine

def should_count_spines(line):
	return line != "" and line != "@"

def determine_num_voices(kern_content):
	max_voices = 0
	line_num = 0
	for line in kern_content.splitlines():
		line = line.rstrip() #right strip
		if (should_count_spines(line)):
			spine_line = SpineLine(line_num, line)
			max_voices = max(max_voices, spine_line.numSpines)
		line_num += 1
	return max_voices

# handles rests and measure numbers
def preprocess(kern_content, max_voices):
	bar = 1
	line_num = 0
	updated_content = ""
	for line in kern_content.splitlines():
		line = line.strip()
		if (line.startswith("@")):
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
	value = value_to_append
	for i in range(1, num_copies):
		value += value_to_append
	return value

def copy_append_staff(num_copies):
	value = ""
	for i in range(0, num_copies):
		value += "*staff{}\t".format(str(i + 1))
	return value

def contains_illegal_operations(line):
	# ref: http://stackoverflow.com/a/3437070
	return "*" in line or "v" in line

def generate_krn(kern_content, output_file_name):
	r = []
	num_voices = determine_num_voices(kern_content)
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
	print('finished writing krn file')

def generate_midi(music_score, output_midi_filename):
	midi_file = music_score.write('midi')
	copyfile(midi_file, output_midi_filename)
	print('finished writing midi file', midi_file)

def generate_xml(music_score, output_xml_filename):
	mxl_file = music_score.write('musicxml')
	copyfile(mxl_file, output_xml_filename)
	print('finished writing musicxml file', mxl_file)

def generate_files(raw_krn_content, output_krn_filename, output_midi_filename, output_xml_filename):
	generate_krn(raw_krn_content, output_krn_filename)
	m = converter.parse(output_krn_filename)
	generate_midi(m, output_midi_filename)
	generate_xml(m, output_xml_filename)

def generate(model_version, checkpoint_name, num_chars, lstm_size, prime="@"):
	# build file paths
	generated_dir = 'generated-music/'
	output_krn_dir = generated_dir + model_version + '/krn'
	output_midi_dir = generated_dir + model_version + '/midi'
	output_xml_dir = generated_dir + model_version + '/xml'
	output_base_filename = 'gen' + str(len(os.listdir(output_krn_dir)) + 1)
	output_krn_filename = output_krn_dir + '/' + output_base_filename + '.krn'
	output_midi_filename = output_midi_dir + '/' + output_base_filename + '.mid'
	output_xml_filename = output_xml_dir + '/' + output_base_filename + '.xml'
	# build checkpoint dir
	checkpoint_dir = 'checkpoints/' + model_version

	print("sampling from model, and generating new music, base file name {}: ".format(output_base_filename))
	raw_krn_content = sample_checkpoint(checkpoint_dir, checkpoint_name, num_chars, lstm_size, prime)
	generate_files(raw_krn_content, output_krn_filename, output_midi_filename, output_xml_filename)
	
if __name__ == "__main__":
	starting_event = "@\n.\t.\t2a/\t.\n"
	generate('v24', '/trained.ckpt', 3000, 2048, starting_event)
