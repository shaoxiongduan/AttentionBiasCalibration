# DEPRECATED


import math
import torch
from timeit import default_timer as timer
from optparse import OptionParser


from config import *
from utility import evaluate_acc, push_ceiling, decode, batch_evaluate_acc
from transformer import Seq2SeqTransformer



usage = "usage: %prog config_path model_path start end/ceil [options]."
parser = OptionParser(usage)
parser.add_option("-g", "--gpu", dest="gpu_id", default='0',
                help="specifies the gpu id to run. values 0 - 7, default 0")
(options, args) = parser.parse_args()
#DEVICE = torch.device(str(DEVICE)+':'+options.gpu_id if torch.cuda.is_available() else 'cpu')
# We don't do this here. It was done in config.py

print('=================')
print(MAX_NUM)
print(FIXED_LEN)
print(BATCH_SIZE)
print(DEVICE)

if len(sys.argv) < 5:
    print('config_path  model_path start end/ceil')
    sys.exit('Too few arguments.')  

model_path = sys.argv[2]
# The 3rd argument should be number of epochs
start = eval(sys.argv[3])
ceiling = False
if sys.argv[4] == 'ceil': 
    ceiling = True
else:
    end = eval(sys.argv[4])


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, VOCAB_SIZE, FFN_HID_DIM)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

checkpoint = torch.load(model_path)
transformer.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epoch']
loss = checkpoint['loss']

# We may want to push the ceiling to see when we 
# Evaluate accuracy
transformer.to(DEVICE)
start_time = timer()
result = 0
prompt = ''
if ceiling:
    print(f"Trying to find ceiling starting {start} on {DEVICE} ..." )
    result = push_ceiling(transformer, start)
    numstr = str(result).rjust(FIXED_LEN, "0")
    print(str(result) + ' -> ' + numstr + ' -> ' + decode(transformer, numstr)[::-1])
    prompt = 'Ceiling'
else:
    print(f"Evaluating between {start} and {end} on {DEVICE} ..." )
    result = batch_evaluate_acc(transformer, BATCH_SIZE, start,  end - start + 1, verbose=True)
    prompt = 'Test acc'

end_time = timer()
elapsed_time = end_time - start_time

print((f"{prompt}:  {result},  elapsed time: {elapsed_time:.3f}s"))

