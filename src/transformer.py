import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer, TransformerDecoderLayer
import math
from typing import Optional, Any, Union, Callable
from torch import functional as F



# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 type: Optional[str] = 'Vanilla',
                 encoding_type: Optional[str] = 'Sinusoidal',
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        
        
        
        pos_embedding = torch.zeros((maxlen, emb_size))
        #if type != 'ALiBi':
        if True:
            pos_embedding[:, 0::2] = torch.sin(pos * den)
            pos_embedding[:, 1::2] = torch.cos(pos * den)

        if encoding_type == 'Sinusoidal':
            pos_embedding = torch.zeros((maxlen, emb_size))
            pos_embedding[:, 0::2] = torch.sin(pos * den)
            pos_embedding[:, 1::2] = torch.cos(pos * den)
        elif encoding_type == 'Log_periodical':
            # Try periodical log positional encoding:
            pos_embedding = torch.ones((maxlen, emb_size))
            for i in range(pos_embedding.size(0)):
                pos_embedding[i, :] = pos_embedding[i, :]*math.log(i%7+1)
        elif encoding_type == 'Sin_periodical':
            pos_embedding = torch.zeros((maxlen, emb_size))
            pos = pos % 3
            pos_embedding[:, 0::2] = torch.sin(pos * den)
            pos_embedding[:, 1::2] = torch.cos(pos * den)
        elif encoding_type == 'NoPE':
            pos_embedding = torch.zeros((maxlen, emb_size))
            # no pos encoding
        else:
            pos_embedding[:, 0::2] = torch.sin(pos * den)
            pos_embedding[:, 1::2] = torch.cos(pos * den)

        # Try log positional encoding:
        #pos_embedding = torch.ones((maxlen, emb_size))
        #for i in range(pos_embedding.size(0)):
        #    pos_embedding[i, :] = pos_embedding[i, :]*math.log(i+1)


        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 custom_encoder: Optional[Any] = None,
                 type: Optional[str] = 'Vanilla',
                 encoding_type: Optional[str] = 'Sinusoidal'):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       custom_encoder=custom_encoder)
        self.generator = nn.Linear(emb_size, vocab_size)
        # The liner project layer, converting from emb_size to vocab_size
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout, type = type, encoding_type = encoding_type)
        

    def forward(self, 
                src: Tensor, 
                tgt: Tensor, 
                src_mask: Optional[Tensor] = None, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
       
        src_emb = self.positional_encoding(self.tok_emb(src))
        tgt_emb = self.positional_encoding(self.tok_emb(tgt))
        #print(src_emb.size())
        #print(tgt_emb.size())
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, memory_mask,
                                src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    #def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        #print(tgt_mask == None)
        return self.transformer.decoder(self.positional_encoding(self.tok_emb(tgt)), memory,
                                        tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)
   
    # Looks like we can use pytorch's Transformer directly


class IdentityEncoder(nn.Module):
    r"""IdentityEncoder is an encoder that simply passess input through. It is used here to 
        implement prefix LM.
    """
    def __init__(self):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
            We have to implement the same interface as TransformerEncoder. But all arguments
            except src are ignored.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product attention.
                Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).
            
        Shape:
            see the docs in Transformer class.
        """
        #return src.clone()
        return src
        # TODO: WHich is right?  
