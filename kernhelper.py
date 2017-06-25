import config
from music21.humdrum.spineParser import SpineLine

def determine_num_voices(kern_content):
    """
    Determines the number of voices present in the generated **kern file
    by scanning each line and determining number of vocies in current line
    relies on music21.humdrum.spineParser.SpineLine class
    """
    max_voices = 0
    line_num = 0
    for line in kern_content:
        line = line.rstrip() #right strip
        if (should_count_spines(line)):
            spine_line = SpineLine(line_num, line)
            max_voices = max(max_voices, spine_line.numSpines)
        line_num += 1
    return max_voices

def should_count_spines(line):
    """
    Returns true if number of spines should be counted for current line
    """
    return line != "" and line != config.MEASURE_SYMBOL

def contains_illegal_operations(line):
    """
    Determines if current line contains operations
    which would likely stump the parsing process, 
	
    More on "Interpretation" lines in **kern: http://www.humdrum.org/guide/ch05/

    ref: http://stackoverflow.com/a/3437070
    """
    return "*" in line or "v" in line