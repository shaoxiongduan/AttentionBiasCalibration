import sys
import torch
from digit_tokenizer  import *
from transformer import IdentityEncoder, Seq2SeqTransformer
import inspect
import seaborn
import os

#FIXME: hardcoded path, injected arguments for ipython, broken notebooks.

#model_path = '/root/workspace/shao/windows_reformatted/exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex/model/epoch-52-loss-0.0080395387-acc-0.0000000000.ckpt' 
#config_path = '/root/workspace/shao/windows_reformatted/exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex/head-8-enc-3-dec-3-emb-128-batch-512-drop-0.3.ini'

#model_path = '/root/workspace/shao/windows_reformatted/exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex/model/epoch-8-loss-0.0150493296-acc-0.0000000000.ckpt'
#config_path = '/root/workspace/shao/windows_reformatted/exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex/head-8-enc-3-dec-3-emb-128-batch-512-drop-0.3.ini'

#model_path = '/root/workspace/shao/github/windows_reformatted_working_version/exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex2/model/epoch-4-loss-0.0053801495-acc-0.0000000000.ckpt'
#config_path = '/root/workspace/shao/github/windows_reformatted_working_version/exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex2/head-8-enc-3-dec-3-emb-128-batch-512-drop-0.3.ini'
#model_path = '/root/workspace/shao/windows_reformatted/exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex2/model/epoch-32-loss-0.0059046340-acc-0.0000000000.ckpt'
#config_path = '/root/workspace/shao/windows_reformatted/exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex2/head-8-enc-3-dec-3-emb-128-batch-512-drop-0.3.ini'
# print(sys.path)
# if not './src' in sys.path:
#     sys.path.append('./src')    # allow the .py scripts to know the path
# print(sys.argv)
# #if not sys.argv[1] == config_path:
# # Don't use this because argv[1] may not exist    
# if not config_path in sys.argv:
#     sys.argv.insert(1, config_path)
#     sys.argv.insert(2, model_path)
# print(sys.argv)
# ipython runs with a bounch of its own arguments but we need to have these so config.py can initialize 
# properly when it is imported. Use insert 'argv.insert' instead of 'argv[1] =' to avoid messing up
# ipython's own arguments. Also since we may run a cell multiple times, we check in case we have aleady
# inserted.

# FIXME: above ipython stuff

# Only import config now. Hopefully we can pass the above sys args to config.py
from config import *
DEVICE = 'cuda:0'

#print("fewaav")
#transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 #NHEAD, VOCAB_SIZE, FFN_HID_DIM)
#optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

#transformer.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epochs = checkpoint['epoch']
#loss = checkpoint['loss']

#transformermodel = torch.load(fullmodel_path)

model_path = sys.argv[2]

transformermodel = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, VOCAB_SIZE, FFN_HID_DIM, custom_encoder=IdentityEncoder(), type = MODEL_TYPE, encoding_type=ENCODING_TYPE)
optimizer = torch.optim.Adam(transformermodel.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

checkpoint = torch.load(model_path)
#print(checkpoint['model_state_dict'])
transformermodel.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epoch']
loss = checkpoint['loss']

transformermodel.to(DEVICE)
transformermodel.eval()


import torch
import seaborn
from utility import batch_decode, batch_evaluate_acc, test_extrapolation, tokenize_nums, get_batch, greedy_decode, decode
import matplotlib.pyplot as plt


print("---------------------")
print("Testing Extrapolation")
IS_EVAL = True
if IS_EVAL:
    
    print('Evaluating ...')
    print(LOAD_WINDOW)
    print(WINDOW_PATH)
    test_extrapolation(transformermodel, 1, 60, 50, "path for extrapolation plot goes here", "path for histogram of wrong digits goes here")

