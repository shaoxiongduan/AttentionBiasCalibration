import math
import pprint
from matplotlib import pyplot as plt
from torch import Tensor
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
import os
from datetime import datetime
import time
import random
from matplotlib.ticker import ScalarFormatter
from config import *
import seaborn
    

def dec_to_base(num,base):  #Maximum base - 36
    base_num = ""
    if num == 0:
        return '0'
    
    while num>0:
        dig = int(num%base)
        if dig<10:
            base_num += str(dig)
        else:
            base_num += chr(ord('A')+dig-10)  #Using uppercase letters
        num //= base
    base_num = base_num[::-1]  #To reverse the string
    return base_num


def base_to_dec(num_str,base):
    num_str = num_str[::-1]
    num = 0
    for k in range(len(num_str)):
        dig = num_str[k]
        if dig.isdigit():
            dig = int(dig)
        else:    #Assuming its either number or alphabet only
            dig = ord(dig.upper())-ord('A')+10
        num += dig*(base**k)
    return num


'''
About masking:

Note that from MultiheadAttention's doc,  "For a float mask, the mask values will be added to the 
attention weight". Positions in the attention matrix with -inf will result in 0s after softmax so 
the corresponding attention will be ignored. We can implement AliBi or windowed attention using 
masking.

When we do cross attention, we use target as query, encoder output as both key and value. During 
inference, target is growing one token at a step so the length is changing. Let L be the current
length of the target, S the length of source (encoder output), the attention matrix W is L x S.
W_ij is the attention weight of q_i over k_j. 

With self attention, attention matrix is computed over source or target only. The encoder works on
source and the decoder works on target. The source (or target) sequence acting as query, key, and 
value. So W is an S x S (encoder) or L x L (decoder) matrix. Encoder has only self attention.

During training, the target is known in full. Therefore both source and target masks are squares.
Source mask is a square mask of all False, allowing all source tokens to be attended to.

We only use AliBi or windowed attention, or any other fancy attention, with decoder, for both self
and cross attentions. According to Transformer's docs, the arguments tgt_mask is used for self 
attention, memory_mask is used for cross attention. Both can be passed to Transformer.forward() 
function. 


These fancy masks are only for decoder!  TODO: we need to make sure that other parts of the code are 
using the same masking! e.g. evaluate()

Windowed attention: all self-attention bias should be a diagonal belt of width window_size of 0s. The 
rest of the elements should be -inf.

Cross attention: should select the most relevant position,

'''


'''
Masking functions for model training and inferencing. It creates an upper triangle matrix of 
-infs:

>>> tgt_mask = generate_square_subsequent_mask(3)
>>> tgt_mask
tensor([[0., -inf, -inf],
        [0., 0., -inf],
        [0., 0., 0.]])
'''

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

'''
Creates a 2-D mask with 0s on the main diagonal and window_size diagonals *below* the main 
diagonal, and -inf elsewhere. Note that anything above the main diaginal remain -inf no matter what 
w is. This is to prevent the model from seeing the future. It won't work otherwise!

>>> mask = generate_windowed_square_mask(6, 1)
>>> mask
tensor([[0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf],
        [-inf, 0., 0., 0., -inf, -inf],
        [-inf, -inf, 0., 0., 0., -inf],
        [-inf, -inf, -inf, 0., 0., 0.]])
'''

#ALiBi stuff

def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head

    * `n_heads` is the number of heads in the attention layer $n$

    The slope for first head is

    $$\frac{1}{2^{\frac{8}{n}}} = 2^{-\frac{8}{n}}$$

    The slopes for the rest of the heads are in a geometric series with a ratio same as above.

    For instance when the number of heads is $8$ the slopes are
    $$\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$$
    """

    # Get the closest power of 2 to `n_heads`.
    # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
    # and then add the remaining slopes.
    n = 2 ** math.floor(math.log2(n_heads))
    # $2^{-\frac{8}{n}}$
    m_0 = 2.0 ** (-8.0 / n)
    # $2^{-1\frac{8}{n}}, 2^{-2 \frac{8}{n}}, 2^{-3 \frac{8}{n}}, \dots$
    m = torch.pow(m_0, torch.arange(1, 1 + n))

    # If `n_heads` is not a power of 2, then we add the remaining slopes.
    # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
    # And pick the slopes upto `n_heads`.
    if n < n_heads:
        # $2^{-\frac{8}{2n}}$
        m_hat_0 = 2.0 ** (-4.0 / n)
        # $2^{-1\frac{8}{2n}}, 2^{-3 \frac{8}{2n}}, 2^{-5 \frac{8}{2n}}, \dots$
        # Note that we take steps by $2$ to avoid slopes added previously.
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        # Concatenate the slopes with the remaining slopes.
        m = torch.cat([m, m_hat])

    return m


def get_alibi_biases(n_heads: int, sz, window_size):
    """
    ## Calculate the attention biases matrix

    * `n_heads` is the number of heads in the attention layer
    * `mask` is the attention mask of shape `[seq_len_q, seq_len_k]`

    This returns a matrix of shape `[seq_len_q, seq_len_k, n_heads, ]` with ALiBi attention biases.
    """ 
    if window_size != -1:
        mask = generate_windowed_square_mask(ATTN_WINDOW, sz).to(DEVICE)
    else:
        mask = generate_square_subsequent_mask(sz).to(DEVICE)
    # Get slopes $m$ for each head
    m = get_slopes(n_heads).to(DEVICE)

    # Calculate distances $[0, 1, \dots, N]$
    # Here we calculate the distances using the mask.
    #
    # Since it's causal mask we can just use $[0, 1, \dots, N]$ too.
    # `distance = torch.arange(mask.shape[1], dtype=torch.long, device=mask.device)[None, :]`
    #print(mask == 0)
    distance = (mask == 0).cumsum(dim=0).to(DEVICE)

    # Multiply them pair-wise to get the AliBi bias matrix
    return (distance[:, :, None] * (-m[None, None, :]) + mask.unsqueeze(2)).to(DEVICE)


def generate_windowed_square_mask(sz, window_size):
    x  = torch.empty(sz, sz)
    x.fill_(-float('inf'))

    # create a mask tensor with 1s on the diagonals to fill
    mask = torch.eye(sz, dtype=torch.bool)
    for i in range(1, window_size+1):
        if sz-i > 0:
            #tmp = torch.diag(torch.ones(sz-i), i) | torch.diag(torch.ones(sz-i), -i)
            #d1 = torch.diag(torch.ones(sz-i), i).type(dtype=torch.bool)
            d2 = torch.diag(torch.ones(sz-i), -i).type(dtype=torch.bool)
            #mask |= d1 | d2
            mask |= d2
            # Do not open the off diagonal elements above the main diagonal because they are preventing
            # the model from seeing the future during training. This maybe necassary because when we 
            # use float mask, the elements are added to the attention and the binary mask mechanism stops
            # working. This way, even when we use a large window size, the upper triangle elements are
            # still -inf so the model won't see the future during training.


    # set the diagonals to 0
    x[mask] = 0
    
    return x.to(DEVICE)

'''
Generates mask to bias the cross attention. Note since we are generating the result sequence
in a reversed order, this mask is a reversed belt of 0s.

Args:
    S: the source sequence length. this is the length of encoder output. (required).
    L: the target sequence length (required).
    window_size: window_size.


Output:

    A tensor of shape [L, S]. For its i-th row, the (S-i) and left and right window_size
    elements are 0s. The rest are -inf

    >>> mask = generate_cross_attention_mask(6, 4, 1)
    >>> mask
    tensor([[-inf, -inf, -inf, -inf, 0., 0.],
        [-inf, -inf, -inf, 0., 0., 0.],
        [-inf, -inf, 0., 0., 0., -inf],
        [-inf, 0., 0., 0., -inf, -inf]])


NOT sure if this is correct. remember we pad to FIXED_LEN!

AND do we do this during training? We are computing loss as a whole??? not auto-regressively!!

'''
def generate_cross_attention_mask(S, L, window_size):
    mask  = torch.empty(L, S)
    mask.fill_(-float('inf'))

    for i in range(L):
        if S >= i:
            pos = S-i-1
            mask[i][pos] = 0
            for j in range(1, window_size+1):
                if pos-j >= 0:
                    mask[i][pos-j] = 0
                #if pos+j < S:
                #    mask[i][pos+j] = 0
    
    #return None    # for testing
    if window_size == -1:
        mask.fill_(0)
    return mask.to(DEVICE)


'''
Creates windowed masks for use with windowed attention, including source, target, amd memory (cross attention)
masks. Note that during inference cross attention mask (memory mask) is dynamically generated but during training
it is generated at once before each epoch. 

Also note that only tgt_mask and memory_mask are windowed. src_mask that is to be used with encoder is the
usual fully visible mask. This is because windowed attention is only for decoder, for now.

'''

def create_windowed_masks(src, tgt, window_size):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_windowed_square_mask(tgt_seq_len, window_size)
    memory_mask = generate_cross_attention_mask(src_seq_len, tgt_seq_len, window_size)


    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    # This is a boolean mask. From docs for MultiHeadAttention: For a binary mask, a ``True`` value indicates 
    # that the corresponding position is not allowed to attend.

    # torch.zeros((src_seq_len, src_seq_len),device=DEVICE) creates a [src_seq_len, src_seq_len] tensor of 
    # all 0's on DEVICE, and .type(torch.bool) turns the element into boolean. In this case, a matrix of all
    # False elements. This allows all source tokens to be attened to.

    # We do NOT use windowed attention on encoder so don't change the original mask

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    # key paddings should be of the same type as attention paddings. So we convert them to float.
    # Maybe we don't actually need them. 

    # From MHA's doc:
    # For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
    #     the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value. 
    src_padding_mask = src_padding_mask.float().masked_fill(src_padding_mask == True, float('-inf')).masked_fill(src_padding_mask == False, float(0.0))
    tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == True, float('-inf')).masked_fill(tgt_padding_mask == False, float(0.0))
    src_padding_mask = src_padding_mask.to(DEVICE)
    tgt_padding_mask = tgt_padding_mask.to(DEVICE)
    # didn't seem to work

    src_padding_mask = tgt_padding_mask = None
    '''
    Since we use fixed len sequence, we don't need padding. In addition, since we use float masks to implement windowed
    attention, if we use binary key paddings, we got warnings:

    /usr/local/python3.9.6/lib/python3.9/site-packages/torch/nn/functional.py:4999: UserWarning: Support for mismatched src_key_padding_mask and src_mask is deprecated. Use same type for both instead.
  warnings.warn(
    /usr/local/python3.9.6/lib/python3.9/site-packages/torch/nn/functional.py:4999: UserWarning: Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.
  warnings.warn(
    
    And setting key paddings to float does not seem to work. So we remove those key paddings to get rid of the warnings.
    
    '''

    return src_mask.to(DEVICE), tgt_mask.to(DEVICE), memory_mask, src_padding_mask, tgt_padding_mask

def create_alibi_masks(src, tgt, nhead, window_size):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = get_alibi_biases(nhead, tgt_seq_len, window_size).permute(2, 0, 1).repeat(BATCH_SIZE, 1, 1).to(DEVICE)
    memory_mask = generate_cross_attention_mask(src_seq_len, tgt_seq_len, window_size)


    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    # This is a boolean mask. From docs for MultiHeadAttention: For a binary mask, a ``True`` value indicates 
    # that the corresponding position is not allowed to attend.

    # torch.zeros((src_seq_len, src_seq_len),device=DEVICE) creates a [src_seq_len, src_seq_len] tensor of 
    # all 0's on DEVICE, and .type(torch.bool) turns the element into boolean. In this case, a matrix of all
    # False elements. This allows all source tokens to be attened to.

    # We do NOT use windowed attention on encoder so don't change the original mask

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    # key paddings should be of the same type as attention paddings. So we convert them to float.
    # Maybe we don't actually need them. 

    # From MHA's doc:
    # For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
    #     the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value. 
    src_padding_mask = src_padding_mask.float().masked_fill(src_padding_mask == True, float('-inf')).masked_fill(src_padding_mask == False, float(0.0))
    tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == True, float('-inf')).masked_fill(tgt_padding_mask == False, float(0.0))
    src_padding_mask = src_padding_mask.to(DEVICE)
    tgt_padding_mask = tgt_padding_mask.to(DEVICE)
    # didn't seem to work

    src_padding_mask = tgt_padding_mask = None
    '''
    Since we use fixed len sequence, we don't need padding. In addition, since we use float masks to implement windowed
    attention, if we use binary key paddings, we got warnings:

    /usr/local/python3.9.6/lib/python3.9/site-packages/torch/nn/functional.py:4999: UserWarning: Support for mismatched src_key_padding_mask and src_mask is deprecated. Use same type for both instead.
  warnings.warn(
    /usr/local/python3.9.6/lib/python3.9/site-packages/torch/nn/functional.py:4999: UserWarning: Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.
  warnings.warn(
    
    And setting key paddings to float does not seem to work. So we remove those key paddings to get rid of the warnings.
    
    '''

    return src_mask.to(DEVICE), tgt_mask.to(DEVICE), memory_mask, src_padding_mask, tgt_padding_mask


def create_load_masks(src, tgt, nhead):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = torch.load(WINDOW_SELF_PATH).to(DEVICE)
    memory_mask = torch.load(WINDOW_PATH).repeat(BATCH_SIZE, 1, 1).to(DEVICE)


    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    # This is a boolean mask. From docs for MultiHeadAttention: For a binary mask, a ``True`` value indicates 
    # that the corresponding position is not allowed to attend.

    # torch.zeros((src_seq_len, src_seq_len),device=DEVICE) creates a [src_seq_len, src_seq_len] tensor of 
    # all 0's on DEVICE, and .type(torch.bool) turns the element into boolean. In this case, a matrix of all
    # False elements. This allows all source tokens to be attened to.

    # We do NOT use windowed attention on encoder so don't change the original mask

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    # key paddings should be of the same type as attention paddings. So we convert them to float.
    # Maybe we don't actually need them. 

    # From MHA's doc:
    # For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
    #     the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value. 
    src_padding_mask = src_padding_mask.float().masked_fill(src_padding_mask == True, float('-inf')).masked_fill(src_padding_mask == False, float(0.0))
    tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == True, float('-inf')).masked_fill(tgt_padding_mask == False, float(0.0))
    src_padding_mask = src_padding_mask.to(DEVICE)
    tgt_padding_mask = tgt_padding_mask.to(DEVICE)
    # didn't seem to work

    src_padding_mask = tgt_padding_mask = None
    '''
    Since we use fixed len sequence, we don't need padding. In addition, since we use float masks to implement windowed
    attention, if we use binary key paddings, we got warnings:

    /usr/local/python3.9.6/lib/python3.9/site-packages/torch/nn/functional.py:4999: UserWarning: Support for mismatched src_key_padding_mask and src_mask is deprecated. Use same type for both instead.
  warnings.warn(
    /usr/local/python3.9.6/lib/python3.9/site-packages/torch/nn/functional.py:4999: UserWarning: Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.
  warnings.warn(
    
    And setting key paddings to float does not seem to work. So we remove those key paddings to get rid of the warnings.
    
    '''

    return src_mask.to(DEVICE), tgt_mask.to(DEVICE), memory_mask, src_padding_mask, tgt_padding_mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
    # This is a boolean mask. From docs for MultiHeadAttention: For a binary mask, a ``True`` value indicates 
    # that the corresponding position is not allowed to attend.

    # torch.zeros((src_seq_len, src_seq_len),device=DEVICE) creates a [src_seq_len, src_seq_len] tensor of 
    # all 0's on DEVICE, and .type(torch.bool) turns the element into boolean. In this case, a matrix of all
    # False elements. This allows all source tokens to be attened to.

    src_padding_mask = (src == PAD_IDX).transpose(0, 1).to(DEVICE)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).to(DEVICE)

    
    
    src_padding_mask = tgt_padding_mask = None


    return src_mask.to(DEVICE), tgt_mask.to(DEVICE), src_padding_mask, tgt_padding_mask


#
# Manipulating numbers and digits
#

# This is where SOS and EOS are added.
def tokenize_num_str(numstr: str) -> Tensor:
    num_tokens = tokenizer(SOS+numstr+EOS)
    token_indices = vocab.lookup_indices(num_tokens)
    return torch.tensor(token_indices)   # Turn list into Tensors

def tokenize_num_str_tolist(numstr: str):
    num_tokens = tokenizer(SOS+numstr+EOS)
    tokens = [i for i in num_tokens]
    return tokens

def tokenize_nums(data: list, reversed: bool = False) -> Tensor:
    """
    Tokenize a list of mumbers and pad them.
    Returns a tensor of the shape [seq_len, len(data)]. 
    Each column is a sequence. All sequences are padded to the longest length in the 
    list with PAD symboles.
    """
    batch = list()
    for num in data:
        numstr = str(num)
        if reversed == True:
            numstr = numstr[::-1]    
        # Using slicing to reverse a string. See https://www.w3schools.com/python/python_howto_reverse_string.asp
        # Note that we only revserse the digits, not SOS and EOS         
        batch.append(tokenize_num_str(numstr))
    
    # padding, make sure the dims are right
    return pad_sequence(batch, padding_value=vocab.lookup_indices([PAD])[0])
    


# Tokenize and left zero pad the number so every sample has equal length
def tokenize_and_zeropad(data: list, fixed_len: int = FIXED_LEN, reversed: bool = False) -> Tensor:
    """
    Tokenize a list of mumbers and pad them
    """
    batch = list()

    for num in data:
        
        #numstr = str(num)
        # Consider base:
        if MODE == 'CNT':
            numstr = dec_to_base(num, BASE)
            numstr = numstr.rjust(fixed_len, "0")
        # TODO: We are trying not to 0-pad. See if it works 
        # Not doing 0-padding does not seem to work, it appears to converge slow - maybe not! 

        if MODE == 'ADD':
            numstr = num.rjust(fixed_len // HALF, "0")
        if MODE == 'MULT':
            numstr = num.rjust(fixed_len, "0")
        if MODE == 'PARITY':
            numstr = num.rjust(fixed_len, "0")
        if MODE == 'PARITY_RECURRENT':
            numstr = num.rjust(fixed_len, "0")
        if reversed == True:
            numstr = numstr[::-1]    
        # Using slicing to reverse a string. See https://www.w3schools.com/python/python_howto_reverse_string.asp
        # Note that we only revserse the digits, not SOS and EOS         
        batch.append(tokenize_num_str(numstr))

    # padding, make sure the dims are right. -- this may not be necessary
    return pad_sequence(batch, padding_value=vocab.lookup_indices([PAD])[0])


# Converts a tensor of token indices into their string representations. 
def indices_to_numstr(data: Tensor, seq_dim: int = 0) -> list: 
    if seq_dim == 0:       
        data = data.transpose(0, 1)
    # Transpose the matrix if each column is a sequence. This is because 
    # Tensor.tolist() that we will use later breaks up the matrix along rows

    numstrs = list()
    for row in data:
        numstrs.append(''.join(vocab.lookup_tokens(row.tolist())))
        # string.join() can join the elements of a list
    return numstrs


#print(tokenize_nums(["12345"]))
#print(tokenize_nums(["12345"], reversed=True))



## 
# 
# get_batch creates the <input, output> pair that training can use. The input i specifies the i-th batch to be made.
# Since we are working on counting, <input, output> are simply two consecutive integers. The function also pads each
# sequence to the length of the longest sample in the batch.
#
# The batch has batch_size sequences with length seq_len, thus the returned data has shape ``[seq_len, batch_size]``.
# Each column of the tensor is a sequence to be processed. Same for both source and target.
#
# This function will be called iteratively for each batch.
# 
# Current we only support sequential sample generation and batching. May try randomized later. 
#
# We don't really need to generate the entire data. Can do one batch at a time
#
#
# The Transformer is designed to take in a full sentence, so an input shorter than the transformer’s input capacity is padded.
# https://chatbotslife.com/language-translation-with-transformers-in-pytorch-ff8b32cf848
#
# Is this true only for batched operations?
##

rand_tot_nums = [*range(MAX_NUM)]

def shuffle_num():
    random.shuffle(rand_tot_nums)


def generate_add_data(max_num, maxDigitLen):
    addnum1 = []
    #print(max_num)
    numlen = int(max_num / maxDigitLen)
    for i in range(maxDigitLen):
        testCnt = min(10 ** (i + 1) - 10 ** i, numlen)
        if testCnt == 10 ** (i + 1) - 10 ** i:
            for j in range(testCnt + 1):
                addnum1.append(j)
        else:
            for j in range(testCnt + 1):
                addnum1.append(random.randint(10 ** i, 10 ** (i + 1)))
    
    while len(addnum1) < max_num:
        addnum1.append(random.randint(1, 10 ** maxDigitLen))

    addnum2 = []
    numlen = int(max_num / maxDigitLen)
    for i in range(maxDigitLen):
        testCnt = min(10 ** (i + 1) - 10 ** i, numlen)
        if testCnt == 10 ** (i + 1) - 10 ** i:
            for j in range(testCnt + 1):
                addnum2.append(j)
        else:
            for j in range(testCnt + 1):
                addnum2.append(random.randint(10 ** i, 10 ** (i + 1)))

    while len(addnum2) < max_num:
        addnum2.append(random.randint(1, 10 ** maxDigitLen))

    random.shuffle(addnum1)
    random.shuffle(addnum2)

    src = []
    tgt = []
    for i in range(max_num):
        src.append((str(addnum1[i]).rjust(FIXED_LEN // 2 - 1, "0") + '+' + str(addnum2[i]).rjust(FIXED_LEN // 2, "0")).rjust(FIXED_LEN, "0"))
        tgt.append(str(addnum1[i] + addnum2[i]).rjust(FIXED_LEN // HALF, '0'))
    return src, tgt


def generate_add_data_bias(max_num, maxDigitLen):
    addnum1 = []
    numlen = int(max_num / maxDigitLen)
    for i in range(maxDigitLen, maxDigitLen):
        testCnt = min(10 ** (i + 1) - 10 ** i, numlen)
        if testCnt == 10 ** (i + 1) - 10 ** i:
            for j in range(testCnt + 1):
                addnum1.append(j)
        else:
            for j in range(testCnt + 1):
                addnum1.append(random.randint(10 ** i, 10 ** (i + 1)))
    
    while len(addnum1) < max_num:
        addnum1.append(random.randint(1, 10 ** maxDigitLen))

    addnum2 = []
    numlen = int(max_num / maxDigitLen)
    for i in range(maxDigitLen, maxDigitLen):
        testCnt = min(10 ** (i + 1) - 10 ** i, numlen)
        if testCnt == 10 ** (i + 1) - 10 ** i:
            for j in range(testCnt + 1):
                addnum2.append(j)
        else:
            for j in range(testCnt + 1):
                addnum2.append(random.randint(10 ** i, 10 ** (i + 1)))

    while len(addnum2) < max_num:
        addnum2.append(random.randint(1, 10 ** maxDigitLen))

    random.shuffle(addnum1)
    random.shuffle(addnum2)

    src = []
    tgt = []
    for i in range(max_num):
        src.append((str(addnum1[i]).rjust(FIXED_LEN // 2 - 1, "0") + '+' + str(addnum2[i]).rjust(FIXED_LEN // 2, "0")).rjust(FIXED_LEN, "0"))
        tgt.append(str(addnum1[i] + addnum2[i]).rjust(FIXED_LEN // HALF, '0'))
    return src, tgt

src_data_add, tgt_data_add = generate_add_data(MAX_NUM, 6)

def generate_singmult_data(max_num, maxDigitLen):
    num1 = []
    numlen = int(max_num / maxDigitLen)
    for i in range(maxDigitLen):
        testCnt = min(10 ** (i + 1) - 10 ** i, numlen)
        if testCnt == 10 ** (i + 1) - 10 ** i:
            for j in range(testCnt + 1):
                num1.append(j)
        else:
            for j in range(testCnt + 1):
                num1.append(random.randint(10 ** i, 10 ** (i + 1)))
    
    while len(num1) < max_num:
        num1.append(random.randint(1, 10 ** maxDigitLen))


    random.shuffle(num1)
    num2 = []
    while len(num2) < max_num:
        num2.append(random.randint(1, 9))



    src = []
    tgt = []
    for i in range(max_num):
        src.append(str(num1[i]).rjust(FIXED_LEN - 2, "0") + '*' + str(num2[i]))
        tgt.append(str(num1[i] * num2[i]).rjust(FIXED_LEN, '0'))
    return src, tgt

src_data_mult, tgt_data_mult = generate_singmult_data(MAX_NUM, 6)

import random
def generate_random_binary_str(strLen):
    ans = ''
    for i in range(strLen):
        tmp = random.randint(0, 1)
        ans += str(tmp)
    return ans

def generate_parity_data(max_num, maxDigitLen):
    # the maxDigitLen must be big to avoid having the same data
    parity_src = []
    numlen = int(max_num / maxDigitLen)
    for i in range(1, maxDigitLen + 1):
        for j in range(numlen):
            parity_src.append(generate_random_binary_str(i))

    while len(parity_src) < max_num:
        parity_src.append(generate_random_binary_str(maxDigitLen + 1))
    
    parity_tgt = []
    for i in parity_src:
        parity_tgt.append(str(i.count('1') % 2))
    return parity_src, parity_tgt

def generate_parity_recurrent_data(max_num, maxDigitLen):
    # the maxDigitLen must be big to avoid having the same data
    parity_src = []
    numlen = int(max_num / maxDigitLen)
    for i in range(1, maxDigitLen + 1):
        for j in range(numlen):
            parity_src.append(generate_random_binary_str(i).rjust(FIXED_LEN, '0'))

    while len(parity_src) < max_num:
        parity_src.append(generate_random_binary_str(maxDigitLen + 1).rjust(FIXED_LEN, '0'))
    
    random.shuffle(parity_src)
    parity_tgt = []
    
    for i in parity_src:
        cnt = 0
        res = ''
        for j in reversed(i):
            cnt = cnt ^ int(j)
            res += str(cnt)
        parity_tgt.append(res.rjust(FIXED_LEN, '0'))
    return parity_src, parity_tgt

src_data_parity, tgt_data_parity = generate_parity_data(MAX_NUM, 18)
src_data_parity_recurrent, tgt_data_parity_recurrent = generate_parity_recurrent_data(MAX_NUM, 18)


def get_batch(batch_size: int, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        batch_size: int, the batch size
        i: int, the i-th batch to be made. with sequential batching, the returned batch includes numbers 
        between i*batch_size to (i+1)*batch_size-1

    Returns:
        tuple (source, target), where both have shape ``[seq_len, batch_size]``
    """
    #src_nums = [*range(i*batch_size, (i+1)*batch_size)]
    #random.shuffle(src_nums)
    #tgt_nums = [src_nums[cnt] + 1 for cnt in range(len(src_nums))]

    src_nums = rand_tot_nums[i * batch_size : (i + 1) * batch_size]
    tgt_nums = [src_nums[cnt] + 1 for cnt in range(len(src_nums))]
    src_batch_data_add = src_data_add[i * batch_size : (i + 1) * batch_size]
    tgt_batch_data_add = tgt_data_add[i * batch_size : (i + 1) * batch_size]
    src_batch_data_mult = src_data_mult[i * batch_size : (i + 1) * batch_size]
    tgt_batch_data_mult = tgt_data_mult[i * batch_size : (i + 1) * batch_size]
    src_batch_data_parity = src_data_parity[i * batch_size : (i + 1) * batch_size]
    tgt_batch_data_parity = tgt_data_parity[i * batch_size : (i + 1) * batch_size]
    src_batch_data_parity_recurrent = src_data_parity_recurrent[i * batch_size : (i + 1) * batch_size]
    tgt_batch_data_parity_recurrent = tgt_data_parity_recurrent[i * batch_size : (i + 1) * batch_size]
    
    # Python does not unpack the result of the range() function so we have to unpack ourselves using *.

    #source = tokenize_nums(src_nums)
    #target = tokenize_nums(tgt_nums, reversed=True)
    # Reverse the target digits
    #print(src_batch_data_add)
    #print(tgt_batch_data_add)
    source = None
    target = None
    if MODE == 'CNT':
    #source = tokenize_and_zeropad(src_nums, reversed=REVERSE_INPUT)
        source = tokenize_and_zeropad(src_nums, reversed=REVERSE_INPUT)
        target = tokenize_and_zeropad(tgt_nums, reversed=True)
    elif MODE == 'ADD':
        source = tokenize_and_zeropad(src_batch_data_add, reversed = REVERSE_INPUT)
        target = tokenize_and_zeropad(tgt_batch_data_add, reversed = True)
    elif MODE == 'MULT':
        source = tokenize_and_zeropad(src_batch_data_mult, reversed = REVERSE_INPUT)
        target = tokenize_and_zeropad(tgt_batch_data_mult, reversed = True)
    elif MODE == 'PARITY':
        source = tokenize_and_zeropad(src_batch_data_parity, reversed = REVERSE_INPUT)
        target = tokenize_and_zeropad(tgt_batch_data_parity, reversed = False)
    elif MODE == 'PARITY_RECURRENT':
        source = tokenize_and_zeropad(src_batch_data_parity_recurrent, reversed = REVERSE_INPUT)
        target = tokenize_and_zeropad(tgt_batch_data_parity_recurrent, reversed = False)
    
    #
    # Note:   
    # 
    # source = tokenize_and_zeropad(src_nums, REVERSE_INPUT) 
    # 
    # does NOT work! This is because tokenize_and_zeropad() takes 3 arguments and the 2nd one 
    # is actually the length the sequence is to be padded to. 
    #

    return source, target


#
# Functions related to model inference, such as decode, evaluation, etc
#

#
# function to generate output sequence using greedy algorithm. Note that this function
# only takes a single tokenized sequence, as src, in the shape of [seq_len, 1]. i.e.,
# a column vector.  
#  
def greedy_decode(model: torch.nn.Module, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    print(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)

        memory_mask = None
        if MODEL_TYPE == 'ALiBi':
            tgt_mask = get_alibi_biases(NHEAD, ys.size(0), ATTN_WINDOW).permute(2, 0, 1).repeat(ys.size(1), 1, 1).to(DEVICE)
        elif LOAD_WINDOW:
            print("loading window")
            tgt_mask = torch.load(WINDOW_SELF_PATH)[:ys.size(0), :ys.size(0)]
            if torch.load(WINDOW_PATH).dim() == 3:
                memory_mask = torch.load(WINDOW_PATH)[:, :ys.size(0), :].repeat(ys.size(1), 1, 1).to(DEVICE)
            else:
                memory_mask = torch.load(WINDOW_PATH)[:ys.size(0), :]
            print(torch.load(WINDOW_PATH).size())
        else:
            if ATTN_WINDOW < 0:
                tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            else:            
                tgt_mask = generate_windowed_square_mask(ys.size(0), ATTN_WINDOW).to(DEVICE)
                memory_mask = generate_cross_attention_mask(memory.size(0), ys.size(0), ATTN_WINDOW)

        #out = model.decode(ys, memory, tgt_mask) 
        out = model.decode(tgt=ys, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)          
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# Decodes a single sequence. Returns a string
def decode(model: torch.nn.Module, src_sentence: str) -> str:
    model.eval()

    src = tokenize_nums([src_sentence])
    num_tokens = src.shape[0]    
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    
    print(FIXED_LEN // HALF)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=FIXED_LEN // HALF, 
                               start_symbol=SOS_IDX).flatten()
    # Note greedy_decode() only takes a single sequence

    return "".join(vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace(SOS, "").replace(EOS, "")



#
# Batch decoding involves a number of ways to invoke the model and its methods. Here is a summary of all we figured out so far:
#
# 1. Calling the model name as a function: e.g., transfomer(...). This uses the `__call__` mechanism: The `__call__` method is used to make 
#    an instance of a class callable like a function. In the implementation of the models, `__call__` is set to 'forward'.
# 
# 2. Besides some actions before and after, nn.Module maps `__call__` to forward(). 
# 
# 3. The Transformer model consists of an encoder and a decoder. A call to forward() bassically passes the source through the encoder, 
#    obtaining the internal representation (the memory argument in the code), which is then, together with target aruments, passed through
#    the decoder.
# 
# 4. Pytorch's implementation of Transformer includes only the encoder and the decoder. It does not include the linear layer that projects 
#    decoder output onto vocabulary space (?). In our subclass, Seq2SeqTransformer, this step is done via the generator which is just 
#    nn.Linear(emb_size, vocab_size). In Seq2SeqTransformer.forward(), the generator is invoked after the decoder output.
#
# 5. What forward() returns is logits. This is also what you get by calling model() because of 2. forward() is useful when you train the 
#    model or evaluate loss when you pass source and target together through the model. I think loss is computed over the forward() output 
#    and the entire target sequence as a whole, not autoregressively. -- may not be the case/
# 
# 6. At inference time, the output sequence is generated token-by-token autoregressively. So you cannot use forward() because we have to 
#    append the previous tokens as the condition for generating the next token (thus autoregressively). Instead, you call encode() to 
#    obtain the internal representation of the source. Then use a tensor to accumulate decoder's output tokens. Then you call decode() 
#    and generator() iteratively. Each time you use the logits to pick the next token, until you see EOS. 
#

def batch_greedy_decode(model, src, max_len=FIXED_LEN // HALF, start_symbol=SOS_IDX, end_symbol=EOS_IDX):
    src = src.to(DEVICE)
    src_seq_len, batch_size = src.size()   # Each column is a sequence
    stop_flag = [False for _ in range(batch_size)]
    finshed = 0

    memory = model.encode(src, None)     
    # encoder outout. We don't need a src mask when all input sequences are of the same length 
    ys = torch.ones(1, batch_size).fill_(SOS_IDX).type(torch.long).to(DEVICE)
    # Decode results. Initialized to SOS symbol. Each column is a sequence

    for t in range(max_len):
        #tgt_mask = (generate_square_subsequent_mask(ys.size(0))
        #                .type(torch.bool)).to(DEVICE)
        # tgt_mask's shape is [len, len] where len is the length of the sequence generated so far.
        # The same mask is used for all sequences in the batch.
        memory_mask = None
        if MODEL_TYPE == 'ALiBi':
            tgt_mask = get_alibi_biases(NHEAD, ys.size(0), ATTN_WINDOW).permute(2, 0, 1).repeat(ys.size(1), 1, 1).to(DEVICE)
        elif LOAD_WINDOW:
            
            if torch.load(WINDOW_PATH).dim() == 3:
                tgt_mask = torch.load(WINDOW_SELF_PATH)[:, :ys.size(0), :ys.size(0)].repeat(ys.size(1), 1, 1).to(DEVICE)
                memory_mask = torch.load(WINDOW_PATH)[:, :ys.size(0), :].repeat(ys.size(1), 1, 1).to(DEVICE)
            else:
                memory_mask = torch.load(WINDOW_PATH)[:ys.size(0), :]
        else:
            if ATTN_WINDOW < 0:
                tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            else:            
                tgt_mask = generate_windowed_square_mask(ys.size(0), ATTN_WINDOW).to(DEVICE)
                memory_mask = generate_cross_attention_mask(memory.size(0), ys.size(0), ATTN_WINDOW)
        # This may only works with equal length batches.
        
        # tgt_mask's shape is [len, len] where len is the length of the sequence generated so far.
        # The same mask is used for all sequences in the batch.
 
        #print(tgt_mask.size())
        #out = model.decode(ys, memory, tgt_mask) 
        out = model.decode(tgt=ys, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)          
        # The shape of decoder output is [seq_len, batch_size, emb_size] where seq_len is the length of the 
        # sequence generated so far. We want to call the generator (i.e., the output layer) to turn them
        # into probabilities over vocabulary so we transpose it so that batch_size is the first dimension 

        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        # generator is just the linear layer. The last dimension must be the input hidden dim, emb_size in our case. 
        # Each row is a probability distribution for a sample.
        #
        # The `[:, -1]` indexing syntax is used to select the last element from the second dimension of the 
        # tensor. Specifically, the `:` before the comma indicates that we want to select all elements along 
        # the first dimension (batch dimension), and the `-1` after the comma indicates that we want to select 
        # the last element along the second dimension (sequence length), which represents the output of the 
        # last time step in the decoder. So each row represents a sample in the batch. This tensor is then 
        # passed to the generator (output layer) of the model to compute the probability distribution over 
        # the vocabulary, which is used to choose the next word in the decoding process.

        _, next_token = torch.max(prob, dim=1)     # find the max of each row
        # next_token has the shape [batch_size], it is NOT a 2-D tensor. We need to convert it into a [1, batch_size] 
        # tensor so it can be cat to ys. The `unsqueeze` function adds a new dimension to the tensor at the specified 
        # position so here we go:
        next_token = next_token.unsqueeze(0) 

        for i in range(batch_size):
            if stop_flag[i] is False:
                if next_token[0][i].item() == end_symbol:
                    finshed += 1
                    stop_flag[i] = True
            if finshed == batch_size:
                break
            
        if finshed == batch_size:
            break

        ys = torch.cat((ys, next_token), dim=0).to(DEVICE)
    return ys


#
# batch_decode takes in a list of intgers or strings. 
#
def batch_decode(model: torch.nn.Module, src: list):
    model.eval()

    src = tokenize_and_zeropad(src)
    #tgt_tokens = batch_greedy_decode3(model, src, max_len = FIXED_LEN, 
    #                           start_symbol=SOS_IDX, end_symbol = EOS_IDX).flatten()
    tgt_tokens = batch_greedy_decode(model, src, max_len = FIXED_LEN, 
                               start_symbol=SOS_IDX, end_symbol = EOS_IDX)
    _, batch_size = src.size()   # Each column is a sequence
    targets = []
    for i in range(batch_size):
        targets.append("".join(vocab.lookup_tokens(list(tgt_tokens[:,i]))).replace(SOS, "").replace(EOS, "")[::-1])

    return targets

#
# Evaluates accuracy using batch decoding. 30+ times faster on my MBP
# Note that if is_index is True, start is the index into rand_tot_nums[] so that you can use shuffled numbers 
#

def batch_evaluate_acc(model: torch.nn.Module, batch_size: int, start: int, 
                       length: int, is_index: bool=False, verbose: bool=False) -> float:
    model.eval()
    correct = 0
    
    batches = math.floor(length/batch_size)
    remains = length % batch_size


    for i in range(batches+1):
        tgt = None
        if i < batches:
            if is_index:
                if MODE == 'CNT':
                    src =  rand_tot_nums[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'ADD':
                    src = src_data_add[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_add[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'MULT':
                    src = src_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY':
                    src = src_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
            
            else:
                if MODE == 'CNT':
                    src = [*range(start+i*batch_size, start+(i+1)*batch_size)]
                if MODE == 'ADD':
                    src = src_data_add[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_add[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'MULT':
                    src = src_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY':
                    src = src_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
        
        else:
            if remains == 0:
                break
            if is_index:
                if MODE == 'CNT':
                    src =  rand_tot_nums[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'ADD':
                    src = src_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'MULT':
                    src = src_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY':
                    src = src_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
            
            else:
                if MODE == 'CNT':
                    src = [*range(start+batches*batch_size, start+batches*batch_size+remains)]
                if MODE == 'ADD':
                    src = src_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'MULT':
                    src = src_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY':
                    src = src_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
        
        if MODE == 'CNT':
            tgt = [s+1 for s in src]
        decoded = batch_decode(model, src)

        for j in range(len(tgt)):
            numstr = str(decoded[j]).rjust(FIXED_LEN, "0")
            if MODE == 'CNT':
                target = dec_to_base(tgt[j], BASE).rjust(FIXED_LEN, "0")
            if MODE == 'ADD':
                target = tgt[j].rjust(FIXED_LEN, "0")
            if MODE == 'MULT':
                target = tgt[j].rjust(FIXED_LEN, "0")
            if MODE == 'PARITY':
                target = tgt[j].rjust(FIXED_LEN, "0")
            if MODE == 'PARITY_RECURRENT':
                target = tgt[j].rjust(FIXED_LEN, "0")[::-1]

            if numstr == target:
                correct = correct + 1
            else:
                if verbose:
                    print(str(src[j]) + " -> " + decoded[j] + " -> " + target)

    return correct/length

def batch_evaluate_acc_analyze(model: torch.nn.Module, batch_size: int, start: int, 
                       length: int, is_index: bool=False, verbose: bool=False) -> float:
    model.eval()
    correct = 0
    
    batches = math.floor(length/batch_size)
    remains = length % batch_size
    digit_wrong = [0 for i in range(FIXED_LEN)]

    for i in range(batches+1):
        tgt = None
        if i < batches:
            if is_index:
                if MODE == 'CNT':
                    src =  rand_tot_nums[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'ADD':
                    src = src_data_add[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_add[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'MULT':
                    src = src_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY':
                    src = src_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
            
            else:
                if MODE == 'CNT':
                    src = [*range(start+i*batch_size, start+(i+1)*batch_size)]
                if MODE == 'ADD':
                    src = src_data_add[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_add[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'MULT':
                    src = src_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY':
                    src = src_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
        
        else:
            if remains == 0:
                break
            if is_index:
                if MODE == 'CNT':
                    src =  rand_tot_nums[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'ADD':
                    src = src_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'MULT':
                    src = src_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY':
                    src = src_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
            
            else:
                if MODE == 'CNT':
                    src = [*range(start+batches*batch_size, start+batches*batch_size+remains)]
                if MODE == 'ADD':
                    src = src_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'MULT':
                    src = src_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY':
                    src = src_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
        
        if MODE == 'CNT':
            tgt = [s+1 for s in src]
        decoded = batch_decode(model, src)

        
        for j in range(len(tgt)):
            numstr = str(decoded[j]).rjust(FIXED_LEN, "0")
            if MODE == 'CNT':
                target = dec_to_base(tgt[j], BASE).rjust(FIXED_LEN, "0")
            if MODE == 'ADD':
                target = tgt[j].rjust(FIXED_LEN, "0")
            if MODE == 'MULT':
                target = tgt[j].rjust(FIXED_LEN, "0")
            if MODE == 'PARITY':
                target = tgt[j].rjust(FIXED_LEN, "0")
            if MODE == 'PARITY_RECURRENT':
                target = tgt[j].rjust(FIXED_LEN, "0")[::-1]

            if numstr == target:
                correct = correct + 1
            else:
                for k in range(len(decoded[j])):
                    if decoded[j][k] != target[k]:
                        digit_wrong[k] += 1
                if verbose:
                    print(str(src[j]) + " -> " + decoded[j] + " -> " + target)

    return correct/length, digit_wrong

def batch_evaluate_acc_get_data(model: torch.nn.Module, batch_size: int, start: int, 
                       length: int, is_index: bool=False, verbose: bool=False) -> float:
    model.eval()
    iscorrect = []
    
    batches = math.floor(length/batch_size)
    remains = length % batch_size


    for i in range(batches+1):
        tgt = None
        if i < batches:
            if is_index:
                if MODE == 'CNT':
                    src =  rand_tot_nums[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'ADD':
                    src = src_data_add[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_add[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'MULT':
                    src = src_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY':
                    src = src_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
            
            else:
                if MODE == 'CNT':
                    src = [*range(start+i*batch_size, start+(i+1)*batch_size)]
                if MODE == 'ADD':
                    src = src_data_add[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_add[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'MULT':
                    src = src_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_mult[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY':
                    src = src_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity[start+i*batch_size:start+(i+1)*batch_size]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
                    tgt = tgt_data_parity_recurrent[start+i*batch_size:start+(i+1)*batch_size]
        
        else:
            if remains == 0:
                break
            if is_index:
                if MODE == 'CNT':
                    src =  rand_tot_nums[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'ADD':
                    src = src_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'MULT':
                    src = src_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY':
                    src = src_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
            
            else:
                if MODE == 'CNT':
                    src = [*range(start+batches*batch_size, start+batches*batch_size+remains)]
                if MODE == 'ADD':
                    src = src_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_add[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'MULT':
                    src = src_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_mult[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY':
                    src = src_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity[start+batches*batch_size:start+batches*batch_size+remains]
                if MODE == 'PARITY_RECURRENT':
                    src = src_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
                    tgt = tgt_data_parity_recurrent[start+batches*batch_size:start+batches*batch_size+remains]
                
        if MODE == 'CNT':
            tgt = [s+1 for s in src]
        decoded = batch_decode(model, src)


        for j in range(len(tgt)):
            numstr = str(decoded[j]).rjust(FIXED_LEN, "0")
            if MODE == 'CNT':
                target = dec_to_base(tgt[j], BASE).rjust(FIXED_LEN, "0")
            if MODE == 'ADD':
                target = tgt[j].rjust(FIXED_LEN, "0")
            if MODE == 'MULT':
                target = tgt[j].rjust(FIXED_LEN, "0")
            if MODE == 'PARITY':
                target = tgt[j].rjust(FIXED_LEN, "0")
            if MODE == 'PARITY_RECURRENT':
                target = tgt[j].rjust(FIXED_LEN, "0")[::-1]

            if numstr == target:
                iscorrect.append(True)
            else:
                iscorrect.append(False)
                
    return iscorrect


def evaluate_acc(model: torch.nn.Module, start: int, length: int, verbose: bool=False) -> float:
    model.eval()
    correct = 0
    for i in range(start, start+length):
        numstr = str(i).rjust(FIXED_LEN, "0")
        if REVERSE_INPUT:
            numstr = numstr[::-1]

        target = str(i+1).rjust(FIXED_LEN, "0")
        decoded = decode(model, numstr)[::-1]    # [::-1] reverses a string
        if decoded == target:
            correct = correct + 1
        else:
            if verbose:
                print(numstr + " -> " + decoded + " -> " + target)

    return correct/length


# See how big we can go without error
def push_ceiling(model: torch.nn.Module, start: int, max: int = sys.maxsize) -> int:
    model.eval()
    target = ''
    for i in range(start, max):
        if i == start:
            numstr = str(i).rjust(FIXED_LEN, "0")
        else:
            numstr = target
        target = str(i+1).rjust(FIXED_LEN, "0")
        if REVERSE_INPUT:
            numstr = numstr[::-1]

        decoded = decode(model, numstr)[::-1]    # [::-1] reverses a string
        if decoded != target:
            return i
        if i % 100000 == 0:
            print("Reached " + i + ": "  + numstr + " -> " + decoded + " -> " + target)
    print('Reached sys.maxsize!')
    return sys.maxsize




def evaluate(model, calcAcc = False, loss_fn = None):
    model.eval()
    losses = 0

    test_batches = math.floor(MAX_NUM*(1-split_ratio)/BATCH_SIZE)
    start = math.ceil(MAX_NUM*split_ratio/BATCH_SIZE)
    correct = 0

    if loss_fn == None:    # Default cross-encrpty
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


    for i in range(start,  start + test_batches):
        #print(f"Evaluating batch   {i}")
        src, tgt = get_batch(BATCH_SIZE, i)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]    # tensor slicing, remove the last row

        memory_mask = None
        if ATTN_WINDOW < 0:
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        elif LOAD_WINDOW: 
            tgt_mask = torch.load(WINDOW_SELF_PATH).to(DEVICE)
            if torch.load(WINDOW_PATH).dim() == 3:
                memory_mask = torch.load(WINDOW_PATH).repeat(BATCH_SIZE, 1, 1).to(DEVICE)
            else:
                memory_mask = torch.load(WINDOW_PATH)
        else:
            src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask = create_windowed_masks(src, tgt_input, ATTN_WINDOW)

        logits = model(src, tgt_input, src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        #src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        #logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        # The forward function of the model passes the inputs through encoder-decoder and the generator. The last
        # step is just the linear projection. This produces the logits, the non-normalized probabilities.

        tgt_out = tgt[1:, :]      # tensor slicing, remove the first row

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        #
        # 1. The `reshape(-1)` method is a shorthand notation to reshape a tensor such that it has only one 
        #    dimension, with the number of elements left to be inferred automatically. 
        #
        # 2. The `shape` attribute is used to get the size of each dimension of a tensor. The `[-1]` index 
        #    can be used to access the size of the last dimension of a tensor.
        #
        # 3. So logits.reshape(-1, logits.shape[-1]) makes the 2nd dimension, the number of columns, the same 
        #    as that of the last dimension of original logits, which is the size of the vocabulary (?).
        #

        losses += loss.item()

        # Also calculate accuracy. Note that in src, each column is a sequence.
        # 
        # The following way of calling greedy_decode does not work. But I know decode()
        # works so lets just use that 
        #   
        #for col in range(src.shape[1]):
        #    src_seq = src[:, col]   # Get each column
        #    num_tokens = len(src_seq)
        #    decoded_tgt_tokens = greedy_decode(model,  src_seq, src_mask, max_len=num_tokens + 5, 
        #                                   start_symbol=SOS_IDX).flatten()
        #    tgt_seq = tgt[:,col]
        #    if decoded_tgt_tokens.equal(tgt_seq):   # Both are tensors
        #        correct = correct + 1

        if calcAcc:
            src_strs = indices_to_numstr(src)
            tgt_strs = indices_to_numstr(tgt)

            for j in range(len(src_strs)):
            
                input = src_strs[j].replace(EOS, "").replace(SOS, "")
                # Note that decode() only works with ''plain numbers'', test samples obtained from get_batch()
                # has SOS and EOS attached. We should remove them before calling decode().
                decoded = decode(model, input)
                target = tgt_strs[j].replace(EOS, "").replace(SOS, "")
                if decoded == target:
                    correct = correct + 1
                else:
                    print(src_strs[j]+" -> " + decoded + " -> " + target)


    return losses / (MAX_NUM*(1-split_ratio)), correct/(MAX_NUM*(1-split_ratio))



# Checkpoint

# Save a check point so that we can pickup training if necessary. In addition to model state, 
# we also need to save optimizer state
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    path: str = ".", epoch: int = 0, loss: float = 999999, acc: float = 0):
      # path must be a directory
      if os.path.exists(path):
            if not os.path.isdir(path):
                  raise ValueError("path must be a directory.")
      else:
            os.mkdir(path)

      # ts = time.time()
      # dt = datetime.fromtimestamp(ts)
      # Or:
      ts = time.localtime()
      dt = time.strftime("%Y-%m-%d-%H-%M-%S", ts)


      #filename = f"{dt}-epoch-{epoch}-loss-{loss:.10f}.ckpt"
      filename = f"epoch-{epoch}-loss-{loss:.10f}-acc-{acc:.10f}.ckpt"
      # This uses Python 3's f-Strings to formart string 
      filepath = os.path.join(path, filename)

      torch.save({  
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, filepath)
      filename = f"full-model-epoch-{epoch}-loss-{loss:.10f}-{acc:.10f}.pt"
      # This uses Python 3's f-Strings to formart string 
      filepath = os.path.join(path, filename)
      torch.save(model, filepath)
      return filepath

def save_list(path: str, data: list):
      with open(path, "w") as f:
            for s in data:
              f.write(str(s) +"\n")


def test_extrapolation(model, startLen, endLen, maxCnt, path, path2):
    acc = []
    xaxs = []
    fig, axs = plt.subplots(endLen - startLen + 1, 1, figsize=(5, 40))
    if MODE == 'CNT':
        for i in range(startLen, endLen + 1):
            
            testCnt = min(10 ** (i + 1) - 10 ** i, maxCnt)
            for j in range(testCnt + 1):
                rand_tot_nums[j] = random.randint(10 ** i, 10 ** (i + 1))
            
            single_acc, wrongcnt = batch_evaluate_acc_analyze(model, batch_size=BATCH_SIZE, start=0, length=testCnt + 1, is_index=True, verbose=False)
            print(f'{round(single_acc, 5)}')
            acc.append(single_acc)
            xaxs.append(f'1e{i}')
        print(acc)
            
    if MODE == 'PARITY':
        for i in range(startLen, endLen + 1):
            print(f"Testing between [1e{i}, 1e{i + 1})")
            testCnt = maxCnt
            for j in range(testCnt + 1):
                src_data_parity[j] = generate_random_binary_str(i)
                tgt_data_parity[j] = str(src_data_parity[j].count('1') % 2)

            single_acc, wrongcnt = batch_evaluate_acc_analyze(model, batch_size=BATCH_SIZE, start=0, length=testCnt + 1, is_index=True, verbose=False)
            print(f'Acc: {single_acc}')
            acc.append(single_acc)
            xaxs.append(f'1e{i}')
    if MODE == 'ADD':
        for i in range(startLen, endLen + 1):
            #print(f"Testing between [1e{i}, 1e{i + 1})")
            testCnt = min(10 ** (i + 1) - 10 ** i, maxCnt)
            for j in range(testCnt + 1):
                a = random.randint(10 ** i, 10 ** (i + 1) - 1)
                b = random.randint(10 ** i, 10 ** (i + 1) - 1)
                src_data_add[j] = ((str(a).rjust(FIXED_LEN // 2 - 1, "0") + '+' + str(b).rjust(FIXED_LEN // 2, "0")).rjust(FIXED_LEN, "0"))
                tgt_data_add[j] = str(a + b).rjust(FIXED_LEN // HALF, '0')

            single_acc, wrongcnt = batch_evaluate_acc_analyze(model, batch_size=BATCH_SIZE, start=0, length=testCnt + 1, is_index=True, verbose=False)
            
            print(f'Acc: {single_acc}')
            acc.append(single_acc)
            xaxs.append(f'{i}')
            
            axs[i - startLen].bar(range(FIXED_LEN), height=wrongcnt, width = 0.5)
            axs[i - startLen].title.set_text(f'Testing between [1e{i}, 1e{i + 1})')
            #seaborn.barplot(x='Digit', y='Wrong cnt', data=wrongcnt, ax=axs[i - startLen])
        print(acc)
    if MODE == 'MULT':
        for i in range(startLen, endLen + 1):
            #print(f"Testing between [1e{i}, 1e{i + 1})")
            testCnt = min(10 ** (i + 1) - 10 ** i, maxCnt)
            for j in range(testCnt + 1):
                a = random.randint(10 ** i, 10 ** (i + 1) - 1)
                b = random.randint(1, 9)
                src_data_mult[j] = ((str(a).rjust(FIXED_LEN - 2, "0") + '*' + str(b)).rjust(FIXED_LEN, "0"))
                tgt_data_mult[j] = str(a * b).rjust(FIXED_LEN, '0')

            single_acc, wrongcnt = batch_evaluate_acc_analyze(model, batch_size=BATCH_SIZE, start=0, length=testCnt + 1, is_index=True, verbose=False)
            
            print(f'Acc: {single_acc}')
            acc.append(single_acc)
            xaxs.append(f'{i}')
            #print("wrongcnt", wrongcnt)
            axs[i - startLen].bar(range(FIXED_LEN), height=wrongcnt, width = 0.5)
            axs[i - startLen].title.set_text(f'Testing between [1e{i}, 1e{i + 1})')
            #seaborn.barplot(x='Digit', y='Wrong cnt', data=wrongcnt, ax=axs[i - startLen])
        print(acc)
    if MODE == 'PARITY_RECURRENT':
        for i in range(startLen, endLen + 1):
            print(f"Testing between [1e{i}, 1e{i + 1})")
            testCnt = maxCnt
            for j in range(testCnt + 1):
                srcres = generate_random_binary_str(i)
                srcres.rjust(FIXED_LEN, '0')
                tgtres = ''
                cnt = 0
                for k in reversed(srcres):
                    cnt = cnt ^ int(k)
                    tgtres += str(cnt)
                src_data_add[j] = srcres
                tgt_data_add[j] = tgtres.rjust(FIXED_LEN, '0')

            single_acc, wrongcnt = batch_evaluate_acc_analyze(model, batch_size=BATCH_SIZE, start=0, length=testCnt + 1, is_index=True, verbose=True)
            print(f'Acc: {single_acc}')
            acc.append(single_acc)
            xaxs.append(f'{i}')
    plt.tight_layout()
    plt.show()

    plt.savefig(path2)

    plt.close()
    fig, ax = plt.subplots()
    ax.plot(xaxs, acc)
    ax.set_title("Extrapolation Acc")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Acc")
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    
    fig.savefig(path)


