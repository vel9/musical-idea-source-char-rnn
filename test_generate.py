from generate import generate_files

base_file_name = 'gen22'
krn_file = base_file_name + '.krn'
test_krn_file = base_file_name + '-test.krn'
output_xml = base_file_name + '.xml'
output_midi = base_file_name + '.mid'

raw_krn_content = open('generated-music/v24/sandbox/' + krn_file,"r").read()
krn_file = 'generated-music/v24/sandbox/' + test_krn_file
xml_file = 'generated-music/v24/sandbox/' + output_xml
midi_file = 'generated-music/v24/sandbox/' + output_midi

generate_files(raw_krn_content, krn_file, midi_file, xml_file)
