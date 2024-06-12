import math
import time
import torch
from torch import Tensor
from timeit import default_timer as timer
import os

from config import *
from config import DROP_OUT, LOAD_PRETRAINED, DEBUG_MODE
from utility import create_alibi_masks, create_windowed_masks, get_batch, create_mask, save_checkpoint, batch_evaluate_acc, evaluate, save_list, decode, shuffle_num

from transformer import Seq2SeqTransformer, IdentityEncoder


##
# The actual training loop
def printDebugMsg(src_mask: Tensor, tgt_mask: Tensor, memory_mask: Tensor, src_padding_mask: Tensor, tgt_padding_mask: Tensor, window_size):
    print(f"window_size:{window_size}")
    print('src_mask:')
    print(src_mask)
    print(src_mask.size())
    print('tgt_mask:')
    print(tgt_mask)
    print(tgt_mask.size())
    print('memory_mask:')
    print(memory_mask)
    print(memory_mask.size())
    print("\n\n")
    print("###################")

if (LOAD_WINDOW):
    tgt_mask_tmp = torch.load(WINDOW_SELF_PATH).to(DEVICE)
    memory_mask_tmp = torch.load(WINDOW_PATH).to(DEVICE)

def train_epoch(model, optimizer, loss_fn, window_size = -1, report_cycle = 128):
    model.train()    # turn on train mode
    losses = 0

    # Determine the number of batches
    train_batches = math.floor(MAX_NUM*split_ratio/BATCH_SIZE)

    #print(f"training_batches: {train_batches}")

    iters = 0
    for i in range(train_batches):
        src, tgt = get_batch(BATCH_SIZE, i)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]      # ??? Why removing the last row?

        memory_mask = None

        #print(tgt_input.size())

        if MODEL_TYPE == 'ALiBi':
            #print('alibi train')
            src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask = create_alibi_masks(src, tgt_input, nhead = NHEAD, window_size=window_size)
        elif LOAD_WINDOW:
            #print("load window train")
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            tgt_mask = tgt_mask_tmp.repeat(BATCH_SIZE, 1, 1).to(DEVICE)
            memory_mask = memory_mask_tmp.repeat(BATCH_SIZE, 1, 1).to(DEVICE)
        else:
            if window_size < 0:
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            else:
                src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask = create_windowed_masks(src, tgt_input, window_size)

        if DEBUG_MODE == True:
            printDebugMsg(src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask, window_size)
 

        logits = model(src, tgt_input, src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)



        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()


    return losses / (MAX_NUM*split_ratio)

# FIXME: unnessary or even wrong comments

#
# The following are attempts to tell config to load parameters from a file. They didn't work. See
# config.py for what actually worked.
#
# Note: Putting those global variables in config.py and accessing them via 
# 'from config import *' only allows you to read them. Any change, e.g., via loading from
# a configuration file does not work. I guess config may be reloaded and the values are
# reset. We need to fix it. maybe passing the config obj to wherever these parameters are needed?????  
# 
#
# configfile = '../config/base-10-max-64k.ini'
#
# Method 1:
#config = configparser.ConfigParser()
#config.read(configfile)
# Data and training config:
#MAX_NUM = eval(config['Data']['MAX_NUM'])              # Max number
#split_ratio = eval(config['Data']['split_ratio'])      # How we split train and test
# Those can be expressions
#BATCH_SIZE = config.getint('Data', 'BATCH_SIZE')
#
# Model parameters
#EMB_SIZE = config.getint('Model', 'EMB_SIZE')
#NHEAD = config.getint('Model', 'NHEAD')
#FFN_HID_DIM = config.getint('Model', 'FFN_HID_DIM')
#NUM_ENCODER_LAYERS = config.getint('Model', 'NUM_ENCODER_LAYERS')
#NUM_DECODER_LAYERS = config.getint('Model', 'NUM_DECODER_LAYERS')

#load_config(configfile)
#print_config()

# Method 2:
#variables = read_config('../config/base-10-max-64k.ini')
#print(variables)
# Update local variables with values from the configuration file
#locals().update(variables)
# Configuraton contains expressions. Must evaluate them, and turn str into numbers for numerical variables

print('=================')
print(f'MAX_NUM: {MAX_NUM}')
print(f'FIXED_LEN: {FIXED_LEN}')
print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'DROP_OUT: {DROP_OUT}')
print(f'ATTN_WINDOW: {ATTN_WINDOW}')
print('=================')

# Use CPU if CUDA not available even if the config specifies cuda
DEVICE = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
#variables['DEVICE'] = DEVICE 

encoder = IdentityEncoder()
#encoder = None
# Try identity encoder. When we do this, all encoder parameters are ignored.

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, VOCAB_SIZE, FFN_HID_DIM, dropout = DROP_OUT, custom_encoder=encoder, type = MODEL_TYPE, encoding_type=ENCODING_TYPE)

torch.manual_seed(0)

for p in transformer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

NUM_EPOCHS = 10


load_model_path = 'pretrained model path goes here'

if LOAD_PRETRAINED == True:
    checkpoint = torch.load(load_model_path)
    transformer.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if FREEZE_EMBEDDING:
        transformer.tok_emb.embedding.weight.requires_grad = False
    transformer.to(DEVICE)
    transformer.train()

if len(sys.argv) >= 3:
     NUM_EPOCHS = int(sys.argv[2])
# The 2nd argument should be number of epochs


train_losses = []
val_losses = []
train_accs = []
val_accs = []

epoch_time = [] 
best_val_loss = math.inf
best_acc = 0
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
split = math.floor(MAX_NUM*split_ratio)
shuffle_num()

ts = time.localtime()
dt = time.strftime("%Y-%m-%d-%H-%M-%S", ts)

#model_path = f'./model-{dt}'
model_path = f'./model'
transformer.to(DEVICE)

print((f"Training for {NUM_EPOCHS} epochs, each having {MAX_NUM*split_ratio/BATCH_SIZE} batches"))
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, loss_fn, ATTN_WINDOW)
    end_time = timer()
    val_loss, val_acc = evaluate(transformer)
    elapsed_time = end_time - start_time
  
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    epoch_time.append(elapsed_time)
    train_acc = 0
    val_acc = 0
    if epoch % 100000 == 0:
        train_acc = batch_evaluate_acc(transformer, batch_size=BATCH_SIZE, start=0, length=split, is_index=True, verbose=False)
        val_acc = batch_evaluate_acc(transformer, batch_size=BATCH_SIZE, start=split+1, length=MAX_NUM-split, is_index=True, verbose=False)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    if val_loss < best_val_loss or val_acc > best_acc:
        best_val_loss = val_loss
        best_acc = val_acc
    save_checkpoint(transformer, optimizer, model_path, epoch, val_loss, val_acc)

    
    print((f"Epoch: {epoch}, Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, Train acc: {train_acc: 6f}, Val acc: {val_acc: 6f}, "f"Epoch time = {elapsed_time:.3f}s"))


save_checkpoint(transformer, optimizer, model_path, epoch, val_loss, val_acc)

save_list(os.path.join(model_path, 'train_losses.txt'), train_losses)
save_list(os.path.join(model_path, 'val_losses.txt'), val_losses)
save_list(os.path.join(model_path, 'train_accs.txt'), train_accs)
save_list(os.path.join(model_path, 'val_accs.txt'), val_accs)


# Evaluate accuracy
numstr = "12345".rjust(FIXED_LEN, "0")
if REVERSE_INPUT:
    numstr = numstr[::-1]

print(decode(transformer, numstr)[::-1])

print('Evaluating ...')
beyond = 10      # check this number of samples beyond MAX_NUM
beyond_acc = batch_evaluate_acc(transformer, batch_size=BATCH_SIZE, start=MAX_NUM, length=beyond, verbose=True)

print((f"beyond: {beyond}, beyond_acc: {beyond_acc:.6f}"))