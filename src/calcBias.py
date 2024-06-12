import sys
import torch
from torch import Tensor
#from vocab import build_vocab_from_iterator
from digit_tokenizer  import *
from transformer import IdentityEncoder, Seq2SeqTransformer
import inspect
import math
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# model_path = './exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120/model/epoch-256-loss-0.0000000097-acc-0.0000000000.ckpt' #replace with interpolated model path
# config_path = './exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120/head-8-enc-3-dec-3-emb-128-batch-512-drop-0.3.ini' #replace with model config
# window_save_path = './exp/plots/' #where to save the windows
# window_plot_path = './exp/window_saves/'


# print(sys.path)
# if not './src' in sys.path:
#     sys.path.append('./src')    # allow the .py scripts to know the path
# print(sys.argv)
#if not sys.argv[1] == config_path:
# Don't use this because argv[1] may not exist    
# if not config_path in sys.argv:
#     sys.argv.insert(1, config_path)
#     sys.argv.insert(2, model_path)
# print(sys.argv)

# ipython runs with a bunch of its own arguments but we need to have these so config.py can initialize 
# properly when it is imported. Use insert 'argv.insert' instead of 'argv[1] =' to avoid messing up
# ipython's own arguments. Also since we may run a cell multiple times, we check in case we have aleady
# inserted.


# Only import config now. Hopefully we can pass the above sys args to config.py
from config  import *
DEVICE = 'cuda:0'

config_path = sys.argv[1]
model_path = sys.argv[2]
window_plot_path = sys.argv[3]
window_save_path = sys.argv[4]

plt.tight_layout()
source = inspect.getsource(torch.nn.functional.multi_head_attention_forward)

replaceval = """need_weights = True
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)"""
new_source = source.replace("tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)", replaceval)
findval = """if is_causal and key_padding_mask is None and not need_weights:"""
replaceval = """if is_causal and key_padding_mask is None and not need_weights:
        print(f'cur: {is_causal and attn_mask == None}\\n\\n')"""

#new_source = new_source.replace(findval, replaceval)
#exec(new_source, torch.nn.functional.__dict__)
#print(new_source)
transformermodel = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, VOCAB_SIZE, FFN_HID_DIM, custom_encoder=IdentityEncoder(), type = MODEL_TYPE)
optimizer = torch.optim.Adam(transformermodel.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

checkpoint = torch.load(model_path, DEVICE)
transformermodel.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epoch']
loss = checkpoint['loss']

transformermodel.to(DEVICE)
transformermodel.eval()


import torch
import seaborn
from utility import batch_decode, batch_evaluate_acc, batch_evaluate_acc_get_data, dec_to_base, test_extrapolation, tokenize_nums, get_batch, greedy_decode, decode, generate_add_data_bias
import matplotlib.pyplot as plt


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

save_output = SaveOutput()
hook_handles = []

#monkey patching to allow pytorch's transformer model to return the attention weights
findval = """return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    merged_mask,
                    need_weights,
                    average_attn_weights,
                    mask_type)
"""

replaceval = """return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    merged_mask,
                    True,
                    average_attn_weights,
                    mask_type)
"""





#source = inspect.getsource(torch.nn.functional.multi_head_attention_forward)
new_source = new_source.replace('attn_output_weights = attn_output_weights.mean(dim=1)', 'attn_output_weights')
new_source = new_source.replace('if need_weights:', 'if True:')
#new_source = new_source.replace('assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"', 'print(f"{is_causal}, {attn_mask}")')

#new_source = new_source.replace('assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"', '')

exec(new_source, torch.nn.functional.__dict__)



#print(new_source)

from torch.nn.modules.transformer import TransformerDecoder

# Replace the original class with the custom subclass
#torch.nn.modules.transformer.TransformerDecoder = CustomTransformerDecoder

print("####################################")

for layer in transformermodel.transformer.decoder.modules():
    if isinstance(layer, torch.nn.modules.activation.MultiheadAttention):
        # print('isinstance')
        handle = layer.register_forward_hook(save_output)
        #print(handle)
        hook_handles.append(handle)

train_batches = math.floor(MAX_NUM*split_ratio/BATCH_SIZE)


iters = 0
correctHistory = []

def dropoffCondition(tgt, mean, stdev) -> bool:
    return tgt < mean + 4.5 * stdev


def dropoffConditionSelf(tgt, mean, stdev) -> bool:
    return tgt < mean + 0.87 * stdev


# This implementation is by no means the most efficient method, but it works and is easily readable
# numstr: source input for the model
# layers: number of layers in the decoder(We are using an identity encoder so there is no need to do this for the encoder)
# M, N: target size of the attention bias matrix
# dir: list of directions we want to scan
# maskSrcLen, maskTgtLen: masking for the zero paddings in the number (set this to the length of numstr)
def generateCrossAttentionBias(numstr: list, layers, M, N, dir: list[tuple[int, int]], maskSrcLen, maskTgtLen):
    attnIn = torch.empty(M, N)
    n = 0
    m = 0
    for cur in numstr:
        curnumstr = cur.rjust(FIXED_LEN, '0')
        out = [i for i in decode(transformermodel, curnumstr)]

        mod_out = save_output.outputs

        last = (len(out) - 1) * 2 * layers + 2 * 2 + 1
        n = mod_out[last][0][1].size(dim = -1)
        m = mod_out[last][0][1].size(dim = 0)
        attnIncur = mod_out[last][0].detach().cpu()
        if cur == numstr[0]:
            attnIn = torch.empty(attnIncur.shape[0], attnIncur.shape[1], attnIncur.shape[2])
        attnIn = torch.add(attnIn, attnIncur)
    attnIn = torch.div(attnIn, len(numstr))

    attnIn = attnIn.numpy()

    H = attnIn.shape[0]
    print(H, M, N)
    attnRes = torch.full((H, M, N), -1e6)
    
    time = len(out) - 1
    fig, axs = plt.subplots(1, H, figsize=(200, 24))

    for h in range(8):
        plt.subplot(1, 8, h + 1)
        sns.heatmap(attnIn[h], vmin=0, cbar=False, square=True)
    plt.show()
    plt.savefig(window_save_path + 'attentionHeatMapCross.png')
    plt.close()
    
    for h in range(H):
        for curdir in dir:
            if curdir[1] > 0:
                attnResTmp = torch.zeros(H, M, N)
                maxval = -1e6
                minval = 1e6

                valSet = set()
                for i in range(M):
                    for j in range(N):
                        tmpi = i
                        tmpj = j
                        size = 0
                        while tmpi - curdir[0] >= 0 and tmpj - curdir[1] >= 0:
                            tmpi -= curdir[0]
                            tmpj -= curdir[1]

                        while tmpi < maskTgtLen and tmpi >= 0 and tmpj < n and tmpj >= 0:
                            attnResTmp[h][i][j] += attnIn[h][tmpi][tmpj]
                            size += 1
                            tmpi += curdir[0]
                            tmpj += curdir[1]
                        if size != 0:
                            if attnResTmp[h][i][j] != 0:
                                valSet.add(attnResTmp[h][i][j].item())
                            maxval = max(maxval, attnResTmp[h][i][j])
                            minval = min(minval, attnResTmp[h][i][j])
                valSet = np.array(list(valSet))

                if valSet.size != 0:
                    q3, q1 = np.percentile(valSet, [75, 25])
                    iqr = q3 - q1
                    mean = np.mean(valSet)
                    stdev = np.std(valSet)
                else:
                    mean = 1e6
                    stdev = 0
                for i in range(M):
                    for j in range(N):
                        if dropoffCondition(attnResTmp[h][i][j], mean, stdev):
                            attnResTmp[h][i][j] = -1e6
            elif curdir[1] < 0:
                attnResTmp = torch.zeros(H, M, N)
                maxval = -1e6
                minval = 1e6
                valSet = set()
                for i in range(M):
                    for j in range(N):
                        tmpi = i
                        tmpj = j
                        size = 0
                        while tmpi - curdir[0] >= 0 and tmpj - curdir[1] < N:
                            tmpi -= curdir[0]
                            tmpj -= curdir[1]
                        while tmpi < maskTgtLen and tmpi >= 0 and tmpj - (N - n) < n and tmpj - (N - n) >= 0:
                            attnResTmp[h][i][j] += attnIn[h][tmpi][tmpj - (N - n)]

                            size += 1
                            tmpi += curdir[0]
                            tmpj += curdir[1]
                        if size != 0:  
                            if attnResTmp[h][i][j] != 0:
                                valSet.add(attnResTmp[h][i][j].item())
                            maxval = max(maxval, attnResTmp[h][i][j])
                            minval = min(minval, attnResTmp[h][i][j])
                valSet = np.array(list(valSet))
                if valSet.size != 0:
                    q3, q1 = np.percentile(valSet, [75, 25])
                    iqr = q3 - q1
                    mean = np.mean(valSet)
                    stdev = np.std(valSet)
                else:
                    mean = 1e6
                    stdev = 0
                for i in range(M):
                    for j in range(N):

                        if dropoffCondition(attnResTmp[h][i][j], mean, stdev):
                            attnResTmp[h][i][j] = -1e6
            else:
                attnResTmp = torch.zeros(H, M, N)
                maxval = -1e6
                minval = 1e6
                valSet = set()
                for i in range(M):
                    for j in range(N):
                        tmpi = i
                        tmpj = j
                        size = 0

                        while tmpi - curdir[0] >= 0 and tmpj - curdir[1] < N:
                            tmpi -= curdir[0]
                            tmpj -= curdir[1]
                        while tmpi < maskTgtLen and tmpi >= 0 and tmpj - (N - n) < n and tmpj - (N - n) >= n - maskSrcLen - 1:
                            attnResTmp[h][i][j] += attnIn[h][tmpi][tmpj - (N - n)]
                            size += 1
                            tmpi += curdir[0]
                            tmpj += curdir[1]
                        if size != 0:  
                            if attnResTmp[h][i][j] != 0:
                                valSet.add(attnResTmp[h][i][j].item())
                            maxval = max(maxval, attnResTmp[h][i][j])
                            minval = min(minval, attnResTmp[h][i][j])
                valSet = np.array(list(valSet))
                if valSet.size != 0:
                    q3, q1 = np.percentile(valSet, [75, 25])
                    iqr = q3 - q1
                    mean = np.mean(valSet)
                    stdev = np.std(valSet)
                else:
                    mean = 1e6
                    stdev = 0
                for i in range(M):
                    for j in range(N):
                        if dropoffCondition(attnResTmp[h][i][j], mean, stdev):
                            attnResTmp[h][i][j] = -1e6
            for i in range(M):
                for j in range(N):
                    attnRes[h][i][j] = max(attnRes[h][i][j], attnResTmp[h][i][j])

            attnRes[h][0][-1] = 0
    return attnRes



# numstr: source input for the model
# layers: number of layers in the decoder(We are using an identity encoder so there is no need to do this for the encoder)
# M, N: target size of the attention bias matrix
# dir: list of directions we want to scan
# maskSrcLen, maskTgtLen: masking for the zero paddings in the number (set this to the length of numstr)
def generateSelfAttentionBias(numstr: list, layers, M, N, dir: list[tuple[int, int]], maskSrcLen, maskTgtLen):
    attnIn = torch.empty(M, N)
    n = 0
    m = 0
    for cur in numstr:

        curnumstr = cur.rjust(FIXED_LEN, '0')
        out = [i for i in decode(transformermodel, curnumstr)]

        mod_out = save_output.outputs
        last = (len(out) - 1) * 2 * layers + 2 * 2
        n = mod_out[last][0][1].size(dim = -1)
        m = mod_out[last][0][1].size(dim = 0)
        attnIncur = mod_out[last][0].detach().cpu()
        if cur == numstr[0]:
            attnIn = torch.empty(attnIncur.shape[0], attnIncur.shape[1], attnIncur.shape[2])
        attnIn = torch.add(attnIn, attnIncur)
    attnIn = torch.div(attnIn, len(numstr))
    attnIn = attnIn.numpy()
    
    H = attnIn.shape[0]
    print(H, M, N)
    attnRes = torch.full((H, M, N), -1e6)

    time = len(out) - 1
    fig, axs = plt.subplots(1, H, figsize=(40, 20))
    for h in range(8):
        plt.subplot(1, 8, h + 1)
        sns.heatmap(attnIn[h], vmin=0, cbar=False, square=True)
    plt.show()
    plt.savefig(window_save_path + 'attentionHeatMapSelf.png')
    plt.close()

    fig, ax = plt.subplots(3, 8, figsize=(200, 24))
    

    cnt = 0
    for h in range(H):
        #print(attnIn[h])
        for curdir in dir:
            if curdir[1] > 0:
                attnResTmp = torch.zeros(H, M, N)
                maxval = -1e6
                minval = 1e6

                valSet = set()
                for i in range(M):
                    for j in range(N):
                        tmpi = i
                        tmpj = j
                        size = 0
                        while tmpi - curdir[0] >= 0 and tmpj - curdir[1] >= 0:
                            tmpi -= curdir[0]
                            tmpj -= curdir[1]

                        while tmpi < maskTgtLen and tmpi >= 0 and tmpj < n and tmpj >= 0:
                            attnResTmp[h][i][j] += attnIn[h][tmpi][tmpj]
                            size += 1
                            tmpi += curdir[0]
                            tmpj += curdir[1]
                        if size != 0:
                            if attnResTmp[h][i][j] != 0:
                                valSet.add(attnResTmp[h][i][j].item())
                            maxval = max(maxval, attnResTmp[h][i][j])
                            minval = min(minval, attnResTmp[h][i][j])
                valSet = np.array(list(valSet))

                if valSet.size != 0:
                    q3, q1 = np.percentile(valSet, [75, 25])
                    iqr = q3 - q1
                    mean = np.mean(valSet)
                    stdev = np.std(valSet)
                else:
                    mean = 1e6
                    stdev = 0
                for i in range(M):
                    for j in range(N):
                        if dropoffConditionSelf(attnResTmp[h][i][j], mean, stdev):
                            attnResTmp[h][i][j] = -1e6
            elif curdir[1] < 0:
                attnResTmp = torch.zeros(H, M, N)
                maxval = -1e6
                minval = 1e6
                valSet = set()
                for i in range(M):
                    for j in range(N):
                        tmpi = i
                        tmpj = j
                        size = 0
                        while tmpi - curdir[0] >= 0 and tmpj - curdir[1] < N:
                            tmpi -= curdir[0]
                            tmpj -= curdir[1]
                        while tmpi < maskTgtLen and tmpi >= 0 and tmpj - (N - n) < n and tmpj - (N - n) >= 0:
                            attnResTmp[h][i][j] += attnIn[h][tmpi][tmpj - (N - n)]

                            size += 1
                            tmpi += curdir[0]
                            tmpj += curdir[1]
                        if size != 0:  
                            if attnResTmp[h][i][j] != 0:
                                valSet.add(attnResTmp[h][i][j].item())
                            maxval = max(maxval, attnResTmp[h][i][j])
                            minval = min(minval, attnResTmp[h][i][j])
                valSet = np.array(list(valSet))
                if valSet.size != 0:
                    q3, q1 = np.percentile(valSet, [75, 25])
                    iqr = q3 - q1
                    mean = np.mean(valSet)
                    stdev = np.std(valSet)
                else:
                    mean = 1e6
                    stdev = 0
                for i in range(M):
                    for j in range(N):

                        if dropoffConditionSelf(attnResTmp[h][i][j], mean, stdev):
                            attnResTmp[h][i][j] = -1e6
            else:
                attnResTmp = torch.zeros(H, M, N)
                maxval = -1e6
                minval = 1e6
                valSet = set()
                for i in range(M):
                    for j in range(N):
                        tmpi = i
                        tmpj = j
                        size = 0

                        while tmpi - curdir[0] >= 0 and tmpj - curdir[1] < N:
                            tmpi -= curdir[0]
                            tmpj -= curdir[1]
                        while tmpi < maskTgtLen and tmpi >= 0 and tmpj - (N - n) < n and tmpj - (N - n) >= n - maskSrcLen - 1:
                            attnResTmp[h][i][j] += attnIn[h][tmpi][tmpj - (N - n)]
                            size += 1
                            tmpi += curdir[0]
                            tmpj += curdir[1]
                        if size != 0:  
                            if attnResTmp[h][i][j] != 0:
                                valSet.add(attnResTmp[h][i][j].item())
                            maxval = max(maxval, attnResTmp[h][i][j])
                            minval = min(minval, attnResTmp[h][i][j])
                valSet = np.array(list(valSet))
                if valSet.size != 0:
                    q3, q1 = np.percentile(valSet, [75, 25])
                    iqr = q3 - q1
                    mean = np.mean(valSet)
                    stdev = np.std(valSet)
                else:
                    mean = 1e6
                    stdev = 0
                for i in range(M):
                    for j in range(N):
                        if dropoffConditionSelf(attnResTmp[h][i][j], mean, stdev):
                            attnResTmp[h][i][j] = -1e6
            for i in range(M):
                for j in range(N):
                    attnRes[h][i][j] = max(attnRes[h][i][j], attnResTmp[h][i][j])
            if torch.max(attnRes[h]) <= -1e6 + 1:
                attnRes[h] = torch.zeros(M, N)
    return attnRes



srcAdd, tgtAdd = generate_add_data_bias(3, 6)

print(srcAdd, tgtAdd)

ansSelf = generateSelfAttentionBias(srcAdd, 3, 62, 62, [(1, 1)], 6, 6)
ansCross = generateCrossAttentionBias(srcAdd, 3, 62, 125, [(1, -1)], 6, 6)


print("finished generating bias, now plotting...")
#code for printing the cross attention mask (self attention is similar)
fig, ax = plt.subplots(1, 8, figsize=(200, 24))
for h in range(8):
    plt.subplot(1, 8, h + 1)
    sns.heatmap(ansCross.detach().cpu().numpy()[h], vmin=-1e6, cbar=False, square=True, annot=True)
plt.show()
plt.savefig(window_save_path + 'extrapolationHeatMapCross.png')
plt.close()


fig, ax = plt.subplots(1, 8, figsize=(200, 24))
for h in range(8):
    plt.subplot(1, 8, h + 1)
    sns.heatmap(ansSelf.detach().cpu().numpy()[h], vmin=-1e6, cbar=False, square=True, annot=True)
plt.show()
plt.savefig(window_save_path + 'extrapolationHeatMapSelf.png')
plt.close()

print(ansCross[0])
print(ansSelf[0])
torch.save(ansCross, window_plot_path + '/windowCross.t')
torch.save(ansSelf, window_plot_path + '/windowSelf.t')
print("done!")


