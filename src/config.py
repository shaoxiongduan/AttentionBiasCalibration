import string
import torch
import sys
from optparse import OptionParser

from vocab import build_vocab_from_iterator
from digit_tokenizer import DigitTokenizer

#
# Configuration and global variables. Here is how this works: other module should import all vaiables 
# (e.g., 'from config import *'). To ensure that all modules sees the same values of those variables.
# The variables are not initialized in the code. Instead, they are read from a config file whose path
# has to be the first argument to the main code. This way, you don't need to call any init or load 
# function for this module (in order to set the variable values according to some config), which won't
# work anyway because other modules that simply import this module and didn't call the functions won't
# see the changes. Instead,  we pass the config file path as an argument and when the module is imported 
# and the code in this file is executed, every importer gets the same values.
#
# The module contains the following variables that are shared among the modules:
# 
# vocalbulary: a list of tokens
# tokenizer: a DigitTokenizer
# vocab: a Vocab object
# parameters read from config: such BASE and model parameters (EMB_SIZE, etc.) 
#

# Read from a config file
import configparser


# configparser stores all variables in low_case. To preserve case, use the following sub-class
class CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr


# Note: configparser only read in strings so you need to convert them to numbers when necessary.
# Also when writing your config file, you don't need to use quotes such as '' or "" for the strings
initialized = False
def load_config(path: str):
    global initialized

    if initialized:
        print("Already initialized, do nothing.")
        return
    
    print("Initializing config from " + path + " ...")

    config = configparser.ConfigParser()
    config.read(path)

    global SOS, EOS, PAD, FIXED_LEN, BASE
    global ADD, SUB, MULT, DIV, INCLUDE_OPS
    global vocalbulary, tokenizer, vocab 
    global PAD_IDX, EOS_IDX, SOS_IDX, VOCAB_SIZE
    global ADD_IDX, SUB_IDX, MULT_IDX, DIV_IDX
    global MAX_NUM, split_ratio, BATCH_SIZE, REVERSE_INPUT
    global EMB_SIZE, NHEAD, FFN_HID_DIM, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROP_OUT, ATTN_WINDOW, LOAD_PRETRAINED
    global DEVICE, MODEL_TYPE, DEBUG_MODE, MODE, ADD_DATA_MAX_LEN, FREEZE_EMBEDDING, ENCODING_TYPE, WINDOW_PATH, LOAD_WINDOW, WINDOW_SELF_PATH
    global HALF
    # Vocabulary and tokenizer
    SOS = config['Vocabulary']['SOS']   # '^'             # Start of a sequence
    EOS = config['Vocabulary']['EOS']   # '$'             # End of a sequence
    PAD = config['Vocabulary']['PAD']   # '@'             # Padding
    INCLUDE_OPS = config.getboolean('Vocabulary', 'INCLUDE_OPS', fallback=False)
    ADD = '+'
    SUB = '-'
    MULT = '*'
    DIV = '/'

    FIXED_LEN = config.getint('Vocabulary', 'FIXED_LEN')   # 10        # We left pad each number with 0s to this length
    BASE = int(config['Vocabulary']['BASE'])

    # Data and training config:
    MAX_NUM = eval(config['Data']['MAX_NUM'])              # Max number
    split_ratio = eval(config['Data']['split_ratio'])      # How we split train and test
    # Those can be expressions
    BATCH_SIZE = config.getint('Data', 'BATCH_SIZE')
    REVERSE_INPUT = config.getboolean('Data', 'REVERSE_INPUT')

    # Model parameters
    EMB_SIZE = config.getint('Model', 'EMB_SIZE')
    NHEAD = config.getint('Model', 'NHEAD')
    FFN_HID_DIM = config.getint('Model', 'FFN_HID_DIM')
    NUM_ENCODER_LAYERS = config.getint('Model', 'NUM_ENCODER_LAYERS')
    NUM_DECODER_LAYERS = config.getint('Model', 'NUM_DECODER_LAYERS')
    DROP_OUT = config.getfloat('Model', 'DROP_OUT', fallback=0.1)
    ATTN_WINDOW = config.getint('Model', 'ATTN_WINDOW', fallback=-1)
    LOAD_PRETRAINED = config.getboolean('Model', 'LOAD_PRETRAINED', fallback=False)
    MODEL_TYPE = config.get('Model', 'MODEL_TYPE', fallback='Vanilla')
    DEBUG_MODE = config.get('Model', 'DEBUG_MODE', fallback=False)
    MODE = config.get('Model', 'MODE', fallback='CNT')
    ADD_DATA_MAX_LEN = config.getint('Model', 'ADD_DATA_MAX_LEN', fallback=5)
    FREEZE_EMBEDDING = config.getboolean('Model', 'FREEZE_EMBEDDING', fallback = False)
    ENCODING_TYPE = config.get('Model', 'ENCODING_TYPE', fallback='Sinusoidal')
    WINDOW_PATH = config.get('Model', 'WINDOW_PATH', fallback='')
    WINDOW_SELF_PATH = config.get('Model', 'WINDOW_SELF_PATH', fallback='')
    LOAD_WINDOW = config.getboolean('Model', 'LOAD_WINDOW', fallback=False)
    HALF = config.getint('Model', 'HALF', fallback=1)
    # Device:
    DEVICE = config['Device']['DEVICE']
    # We also need to parse the commandline options for GPU id
    if '-g' in sys.argv or '-gpu' in sys.argv:
        usage = "usage: %prog config_path model_path start end/ceil [options]."
        parser = OptionParser(usage)
        parser.add_option("-g", "--gpu", dest="gpu_id", default='0',
                    help="specifies the gpu id to run. values 0 - 7, default 0")
        if '-g' in sys.argv:
            i = sys.argv.index('-g')
        else:
            i = sys.argv.index('-gpu')
        (options, args) = parser.parse_args(sys.argv[i:i+2])
        # only parse this part otherwise OptionParser may complain about unsupported options

        DEVICE = str(DEVICE)+':'+options.gpu_id
        
    DEVICE = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    initialized = True

# Build the vocabulary:
def build_vocabulary(base: int) :
    alphanumeric = [*range(0, 10)]                 # This is a list of int.
    # Python does not unpack the result of the range() function so we have to unpack ourselves using *.
    alphanumeric = list(map(str, alphanumeric))    
    # Convert to strings. The map() function applies the given function, here str() to all the elements in alphanumeric
    alphanumeric = alphanumeric + list(string.ascii_uppercase)    # + operator simply combines the two lists
    vocalbulary = alphanumeric[:base]    # Vocabulary is the first BASE symbols. We assume BASE is smaller than 10+26
    vocalbulary.append(EOS)
    vocalbulary.append(SOS)
    vocalbulary.append(PAD) 
    if INCLUDE_OPS:
        vocalbulary.append(ADD) 
        vocalbulary.append(SUB)
        vocalbulary.append(MULT) 
        vocalbulary.append(DIV)    
    
    return vocalbulary


# Load parameters from config file and setup the vocabulary.
load_config(sys.argv[1])
# FIXME: the way parameters are passed are mot so elegent

vocalbulary = build_vocabulary(BASE)
tokenizer = DigitTokenizer()
vocab = build_vocab_from_iterator(map(tokenizer, vocalbulary))

# We can treat our vocalbulary as text data and still use build_vocab_from_iterator(). 
# The term frequencies are not relevant.
PAD_IDX = vocab.lookup_indices([PAD])[0] 
EOS_IDX = vocab.lookup_indices([EOS])[0] 
SOS_IDX = vocab.lookup_indices([SOS])[0]
if INCLUDE_OPS:
    ADD_IDX = vocab.lookup_indices([ADD])[0]
    SUB_IDX = vocab.lookup_indices([SUB])[0]
    MULT_IDX = vocab.lookup_indices([MULT])[0]
    DIV_IDX = vocab.lookup_indices([DIV])[0]

VOCAB_SIZE = len(vocalbulary)


# Configuration stuff
def print_config():
    print(30*'=')
    print((f"BASE: {BASE}, FIXED_LEN: {FIXED_LEN}, SOS: {SOS}, EOS: {EOS}, PAD: {PAD}"))
    print((f"Vocalbulary: {vocalbulary}")) 
    print((f"Vocalbulary: {vocalbulary}"))
    print((f"Looked-up tokens: {vocab.lookup_tokens([*range(0, VOCAB_SIZE)])}"))
    print((f"Token indices: {vocab.lookup_indices(vocalbulary)}"))
    print((f"MAX_NUM: {MAX_NUM}, split_ratio: {split_ratio}, BATCH_SIZE: {BATCH_SIZE}, REVERSE_INPUT: {REVERSE_INPUT}"))
    print((f"EMB_SIZE: {EMB_SIZE}, NHEAD: {NHEAD}, FFN_HID_DIM: {FFN_HID_DIM}, NUM_ENCODER_LAYERS: {NUM_ENCODER_LAYERS}, \
           NUM_DECODER_LAYERS: {NUM_DECODER_LAYERS}, DROP_OUT: {DROP_OUT}, ATTN_WINDOW: {ATTN_WINDOW}"))
    print(f"DEVICE: {DEVICE}")
    print(f"FREEZE_EMBEDDING: {FREEZE_EMBEDDING}")
    print(f"POSITION_ENCODING: {ENCODING_TYPE}")


print_config()

#
# The following is another way of sharing variables. It reads in the variables and builds dict.
# The caller can update its local variables with values from the configuration file by 
# 
#      locals().update(variables)
#
# We are not using this method


import re

def is_expression(s: str):
    # Define a regular expression pattern that matches only digits and the allowed symbols
    pattern = r'^[0-9+\-*/()]+$'

    # Use the search() method to check if the string matches the pattern
    match = re.search(pattern, s)

    # Return True if the pattern matches the entire string, False otherwise
    return match and match.group() == s



def read_config(path: str):
    config = CaseSensitiveConfigParser()
    #config.read('../config/base-10-max-64k.ini')
    config.read(path)

    # Define variables based on values in the configuration file
    sections = config.sections()
    variables = dict()
    for section in sections:
        print(section)
        print(config.options(section))
        variables.update({name: config.get(section, name) for name in config.options(section)})
    
    for key, val in variables.items():
        if is_expression(val):
            variables[key] = eval(val)


    # Return a dictionary containing the variables
    return variables




if __name__ == "__main__":
    # Test config:
    print_config()
    load_config('../config/base-10-max-64k.ini')
    print(30*'=')
    print_config()
    load_config('../config/base-10-max-64k.ini')
    print_config()

    variables = read_config('../config/base-10-max-64k.ini')
    print(variables)
