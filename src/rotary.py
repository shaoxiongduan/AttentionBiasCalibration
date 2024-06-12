from typing import Callable, Optional, Tuple, Union
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer
from torch.nn.modules.activation import MultiheadAttention

from torch.nn import functional as F
# Note that the `relu` function is located in the `nn.functional` module, not the `torch.functional` module.


'''
Transformer with Rotary Positional Embeddings (RoPE)

Part of the code adopted from

https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/rope/__init__.py


Currently it assumes that the sequences are columns of the input tensor so 'batch_first' does not work

'''


class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Adaopted from the above link. Changes:

    The original implementation assumes input tensor to be of the shape `[seq_len, batch_size, n_heads, d]`
    We changed it to `[seq_len, batch_size, embed_dim]` to be consistent with torch's MHA



    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.

    ### For a pair of features

    Let $x^{(1)}_m$ and $x^{(2)}_m$ be two features of the
    key or query of any head at position $m$.
    Or for simplicity assume $x$ has only two features.
    Then the transformation is,

    \begin{align}
    RoPE\big(x^{(1)}_m, x^{(2)}_m, m\big) &=
    \begin{pmatrix}
    \cos m \theta & - \sin m \theta \\
    \sin m \theta & \cos m \theta
    \end{pmatrix}
    \begin{pmatrix}
    x^{(1)}_m \\
    x^{(2)}_m \\
    \end{pmatrix} \\
    &=
    \begin{pmatrix}
    x^{(1)}_m \cos m\theta - x^{(2)}_m \sin m \theta \\
    x^{(2)}_m \cos m\theta + x^{(1)}_m \sin m \theta \\
    \end{pmatrix} \\
    \end{align}

    where $\theta$ is a constant angle. The other pairs of features are transformed similarly.

    ### Attention is relative

    For a pair of features, dot-product attention score between two positions $m$ and $n$ would be

    \begin{align}
    \Big \langle RoPE\big(x^{(1)}_m, x^{(2)}_m, m\big),  RoPE\big(x^{(1)}_n, x^{(2)}_n, n\big) \Big \rangle &= \\
    (x^{(1)}_m \cos m\theta - x^{(2)}_m \sin m \theta)(x^{(1)}_n \cos n\theta - x^{(2)}_n \sin n \theta) &+ \\
    (x^{(2)}_m \cos m\theta + x^{(1)}_m \sin m \theta)(x^{(2)}_n \cos n\theta + x^{(1)}_n \sin n \theta) &= \\
    x^{(1)}_m x^{(1)}_n (\cos m\theta \cos n\theta + \sin m \theta \sin n \theta) &+ \\
    x^{(1)}_m x^{(2)}_n (-\cos m\theta \sin n\theta + \sin m \theta \cos n \theta) &+ \\
    x^{(2)}_m x^{(1)}_n (-\sin m\theta \cos n\theta + \cos m \theta \sin n \theta) &+ \\
    x^{(2)}_m x^{(2)}_n (\sin m\theta \sin n\theta + \cos m \theta \cos n \theta) &= \\

    x^{(1)}_m x^{(1)}_n \cos (m - n) \theta +
    x^{(1)}_m x^{(2)}_n \sin(m - n) \theta &+ \\
    - x^{(2)}_m x^{(1)}_n \sin (m - n) \theta +
    x^{(2)}_m x^{(1)}_n \cos (m - n) \theta &= \\

    \big(x^{(1)}_m \cos (m - n)\theta - x^{(2)}_m \sin (m - n) \theta\big) x^{(1)}_n &+ \\
    \big(x^{(2)}_m \cos (m - n)m\theta + x^{(1)}_m \sin (m - n) \theta\big) x^{(2)}_n  &= \\

    \Big \langle RoPE\big(x^{(1)}_m, x^{(2)}_m, m - n\big),  RoPE\big(x^{(1)}_n, x^{(2)}_n, 0\big) \Big \rangle
    \end{align}

    This shows that for dot-production attention the rotary encodings gives relative attention.

    ### For all features

    The features are grouped into pairs and handled as above. They use a different $\theta$ for each pair.

    The paper suggests using $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    for the $\frac{d}{2}$ pairs of features.

    We pair feature $i$ with feature $i + \frac{d}{2}$. So for position $m$ we transform

    \begin{align}
    \begin{pmatrix}
    x^{(i)}_m \\
    x^{(i + \frac{d}{2})}_m
    \end{pmatrix}
    \end{align}

    to

    \begin{align}
    \begin{pmatrix}
    x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
    x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
    \end{pmatrix} \\
    \end{align}
    """

    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        """
        Cache $\cos$ and $\sin$ values

        This function assumes that the sequences are columns of the input tensor so 'batch_first' does not work.
        And it builds cache of the sin and cos values up to 'x.shape[0]' (sequence length) position. For each 
        position, the values are computed for the full feature length, up to 'self.d'. 

        So the cache is of the shape '[seq_len, batch_size, self.d]'

        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]
        # Currently it assumes that the sequences are columns of the input tensor so 'batch_first' does not work


        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        # Length: self.d/2


        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        #self.cos_cached = idx_theta2.cos()[:, None, None, :]
        #self.sin_cached = idx_theta2.sin()[:, None, None, :]
        self.cos_cached = idx_theta2.cos()[:, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, :]
        # Changed the shape to `[seq_len, batch_size, embed_dim]` to be consistent with torch's MHA


    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        #return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)
        return torch.cat([-x[:, :, d_2:], x[:, :, :d_2]], dim=-1)
        # Changed the shape to `[seq_len, batch_size, embed_dim]` to be consistent with torch's MHA


    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        self._build_cache(x)
        # Why don'y you build cache with x_rope? -- because _build_cache(x) itself only computes up to self.d.
        # The cache is of the shape '[seq_len, batch_size, self.d]

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        # Note that self.d is a fraction of the entire feature dimension
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        # Calculate
        #
        # \begin{align}
        # \begin{pmatrix}
        # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
        # \end{pmatrix} \\
        # \end{align}
        #
        # for $i \in {1, 2, ..., \frac{d}{2}}$
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        # Note the cache is of the shape '[max_seq_len, batch_size, self.d] and here we are only slicing up 
        # to 'x.shape[0]' positions

        # a[:d] is slicing along the first dimension! In out case, the sequence length


        '''
        seq_len = x.shape[0]
        #x_rope = (x_rope * self.cos_cached[:seq_len]) 
        print(self.d)
        print(seq_len)
        print(x_rope.shape)
        print(x_pass.shape)
        print(neg_half_x.shape)
        print(self.cos_cached.shape)
        print(self.cos_cached[:self.d].shape)

        # is this [:seq_len] the right way to select cached suff? this select along the first dimension!!

        print(self.cos_cached[:seq_len].shape)

        x_rope = (x_rope * self.cos_cached[:self.d]) 
        # x_rope is sliced with length self.d (see earlier, it does not match 'seq_len')
        x_rope = x_rope + (neg_half_x * self.sin_cached[:seq_len])

        '''

        return torch.cat((x_rope, x_pass), dim=-1)



class RotaryPEMultiHeadAttention(MultiheadAttention):
    """
    ## Multi-head attention with rotary positional embeddings

    This is adopted from 

    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/rope/__init__.py

    with the following changes:
    
    1. The super class is changed to torch's MultiheadAttention because we will replace the original MHA in 
       torch's Transformer with this one.
    
    2. We do not repy on labmlai's implementation for computing the attention. In stead, we just rotate the key and query
       and use torch's own MHA computation. Note that this deviates from RoPE's paper which uses linear attention (torch's
       MHA uses softmax) but it should work.

    3. Aligned the arguments. Note that the current implementation assumes that the sequences are columns of the input tensor 
       so 'batch_first' does not work

    
    Why do you need number of heads here???
       
    """

    #def __init__(self, heads: int, d_model: int, rope_percentage: float = 0.5, dropout_prob: float = 0.0):
    # labmiai's interface

    def __init__(self, embed_dim, num_heads, dropout=0.3, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, rope_percentage=0.5) -> None:
        if batch_first:
            raise RuntimeError("batch_first does not work with RotaryPEMultiHeadAttention!")

        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                 kdim, vdim, batch_first, device, dtype)

        # Rotary positional embedding layers
        self.d_k = embed_dim
        d_rope = int(self.d_k * rope_percentage)
        self.query_rotary_pe = RotaryPositionalEmbeddings(d_rope)
        self.key_rotary_pe = RotaryPositionalEmbeddings(d_rope)


        # Is this right? doesn't the embedding require to know the whole feature dimension? or is it just comput theta's with 
        # this fraction?



    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys

        This may not be useful in our case

        """

        # Calculate dot-product with RoPE
        return torch.einsum('ibhd,jbhd->ijbh', self.query_rotary_pe(query), self.key_rotary_pe(key))
    

    '''
      Copied from torch's MHA so the arguments match:
      
      NOTE: The shapes may not match!!  RotaryPositionalEmbeddings seems to require heads explicitly


    '''
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        
        """
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
            Default: ``False``.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """

        rope_query = self.query_rotary_pe(query)
        rope_key = self.key_rotary_pe(key)

        # That's it?

        return super().forward(
                    query = rope_query,
                    key = rope_key,
                    value = value,
                    key_padding_mask = key_padding_mask,
                    need_weights = need_weights,
                    attn_mask = attn_mask,
                    average_attn_weights = average_attn_weights,
                    is_causal = is_causal
                    )

def _test_rotary():
    """
    Testing RoPE with a simple example
    """
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float)
    #x = x[:, None, None, :]
    x = x[:, None, :]
    # Changed the shape to `[seq_len, batch_size, embed_dim]` to be consistent with torch's MHA

    # Why adding these two dimension? we have to get it work with 3-d tensors: [seq_len, batch_size, d]`
    # the original implementation requires [seq_len, batch_size, n_heads, d]`

    emb_dim = 16
    batch_size = 2
    seq_len = 8
    #    x = torch.rand(seq_len, batch_size, emb_dim)

    #print(x)

    rotary_pe = RotaryPositionalEmbeddings(4)
    # The dimension passed to RotaryPositionalEmbeddings better be dividable by 2

    # Test: pick two vectors, put them into different positions but making sure they are k positions apart.
    # Their dot product should remain the same.
    # 
    # x is composed of two vectors x1 and x2
    x = torch.rand(2, batch_size, emb_dim)
    x = torch.cat((x, x, x, x, x, x), dim=0)
    print(x.shape)
    x_rope = rotary_pe(x) 

    distance = 3
    seq_len = x.shape[0]
    for i in range(0, seq_len-distance, distance+1):
        x1 = x_rope[i, 0,:]
        x2 = x_rope[i+distance, 0,:]
        print(torch.dot(x1, x2))


if __name__ == '__main__':
    _test_rotary()


'''
Inherited from pytorch's nn.Module.TransformerDecoderLayer and hacked to implement the rotary positional encoding.

Usage:
    Create a TransformerDecoder with this decoder layer, and create the transformer model with it.

'''

class RoPEDecoderLayer(TransformerDecoderLayer):
    r""" Everything is the same except for MHA
    
    
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, 
                         batch_first, norm_first, device, dtype)
        '''
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        '''
        # Replace the original MHA with RoPEMultiheadAttention:
        #factory_kwargs = {'device': device, 'dtype': dtype}
        #self.self_attn = RotaryPEMultiHeadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first,
        #                                    **factory_kwargs)
        #self.multihead_attn = RotaryPEMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                         **factory_kwargs)

        self.self_attn = RotaryPEMultiHeadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = RotaryPEMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # should we do cross attention?


        # And we are done ?!
