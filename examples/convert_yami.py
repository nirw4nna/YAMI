import torch
import os
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
import struct
from enum import Enum
from dataclasses import dataclass, fields
import abc
import ctypes


# Format specification for the tokenizer and the model files.
# All the numbers are in little-endian.
#
# yami_tokenizer.bin format:
#   - header (5 bytes) = 0x59 0x41 0x4D 0x49 0x54 ("YAMIT" as hex string)
#   - file version (1 byte)
#   - tokenizer type (1 byte: 0=BPE, 1=SP)
#   - if SP:
#       -- vocab size (4 bytes)
#       -- foreach vocab:
#           --- size (2 bytes)
#           --- vocab (size bytes, UTF-8)
#           --- score (4 bytes, float)
#   - if BPE:
#       -- encoder size (4 bytes)
#       -- foreach token:
#           --- size (2 bytes)
#           --- token (size bytes, UTF-8)
#       -- bpe merges size / vocab (4 bytes)
#       -- foreach bpe pair:
#           --- size (2 bytes)
#           --- bpe pair (size bytes, two UTF-8 strings separated by whitespace)
#
# yami_model.bin format:
#   - header (5 bytes) = 0x59 0x41 0x4D 0x49 0x4D ("YAMIM" as hex string)
#   - file version (1 byte)
#   - model type (1 byte: 0=GPT-2, 1=LLaMA)
#   - hparams size (2 bytes)
#   - hparams
#   - number of tensors (2 bytes)
#   - foreach tensor:
#       -- label (64 bytes, ASCII)
#       -- encoding (1 byte: 0=fp32)
#       -- number of dimensions (1 byte)
#       -- dimensions (8 bytes per dimension, always 4 dimensions)
#       -- data offset wrt. the beginning of this file (8 bytes)
#   - foreach tensor:
#       -- data (size bytes in row-major order, encoded as specified)

TOKENIZER_HEAD = b'\x59\x41\x4D\x49\x54'
MODEL_HEAD = b'\x59\x41\x4D\x49\x4D'

_FILE_VERSION = 1

_TYPE_MAP = {
    int: ctypes.c_uint32,
    float: ctypes.c_float
}

# Base class for the hparams of all the model, used for export
@dataclass
class Hparams(abc.ABC):
    
    @classmethod
    def to_raw(cls) -> bytearray:
        struct_fields = [(field.name, _TYPE_MAP[field.type]) for field in fields(cls)]
        struct_class = type('YamiHparams', (ctypes.Structure, ), {'_fields_': struct_fields})
        the_struct = struct_class(*[getattr(cls, field) for field, _ in struct_fields])
        return bytearray(the_struct)


class Tokenizer(Enum):
    BPE = 0
    SP = 1

    def __str__(self) -> str:
        return f'{self.name}'
    

class Model(Enum):
    GPT2 = 0
    LLAMA = 1

    def __str__(self) -> str:
        return f'{self.name}'

def _map_location(storage, location):
    return storage

def load_from_meta(checkpoint_folder: str):
    with open(os.path.join(checkpoint_folder, 'params.json')) as f:
        params = json.load(f)
    
    checkpoint_path = list(Path(checkpoint_folder).glob('consolidated.*.pth'))[0]
    return torch.load(checkpoint_path, map_location=_map_location), params
    

def _export_bpe(encoder, vocab, file_handle):
    file_handle.write(struct.pack('<I', len(encoder)))
    for k in encoder:
        raw = bytes(k, encoding='utf-8')
        file_handle.write(struct.pack('<H', len(raw)))
        file_handle.write(raw)
    
    file_handle.write(struct.pack('<I', len(vocab)))
    for k in vocab:
        bpe_pair = ' '.join(k)
        raw = bytes(bpe_pair, encoding='utf-8')
        file_handle.write(struct.pack('<H', len(raw)))
        file_handle.write(raw)


def _export_sp(in_model, file_handle):
    sp = SentencePieceProcessor(model_file=in_model)
    n_words = sp.vocab_size()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    file_handle.write(struct.pack('<I', n_words))
    for i in range(n_words):
        token = sp.id_to_piece(i)
        score = sp.get_score(i)
        
        if i == bos_id:
            token = '<s>'
        if i == eos_id:
            token = '</s>'

        token = token.replace('▁', ' ')
        raw_token = bytes(token, encoding='utf-8')
        file_handle.write(struct.pack('<H', len(raw_token)))
        file_handle.write(raw_token)
        file_handle.write(struct.pack('<f', score))


def export_tokenizer(out_file: str, type: Tokenizer,
                     in_model: str = None, encoder=None,
                     vocab=None):
    print(f'Exporting {type} tokenizer to "{out_file}"')

    with open(out_file, mode='wb') as f:
        f.write(TOKENIZER_HEAD)
        f.write(struct.pack('<B', _FILE_VERSION))
        f.write(struct.pack('<B', type.value))
        if type == Tokenizer.SP:
            _export_sp(in_model=in_model, file_handle=f)   

        if type == Tokenizer.BPE:
            _export_bpe(encoder=encoder, vocab=vocab, file_handle=f)   


def export_model(out_file: str, model: Model, model_dict, hparams: Hparams, to_transpose=None):
    print(f'Exporting {model} model with {len(model_dict)} tensors to "{out_file}"')
    
    with open(out_file, mode='wb') as f:
        f.write(MODEL_HEAD)
        f.write(struct.pack('<B', _FILE_VERSION))
        f.write(struct.pack('<B', model.value))
        raw_hparams = hparams.to_raw()
        f.write(struct.pack('<H', len(raw_hparams)))
        f.write(raw_hparams)

        # number of tensors
        f.write(struct.pack('<H', len(model_dict)))
        # 106 is the tensor metadata size
        offset = len(MODEL_HEAD) + 6 + len(raw_hparams) + len(model_dict) * 106
        for label in model_dict:
            tensor = model_dict[label]
            print(f'Found new tensor: [ {label} {tensor.shape} {tensor.dtype} ]')
            if to_transpose and any([label.endswith(tt) for tt in to_transpose]):
                print(f'Tensor {label} will be transposed')
                tensor = tensor.t()

            model_dict[label] = tensor

            # Write all the tensors metadata
            label_len = len(label)
            if label_len >= 64:
                raise Exception(f'Label {label} is too long')

            f.write(bytes(label, encoding='ascii') + (b'\x00' * (64 - label_len)))

            if not tensor.dtype == torch.float32:
                tensor = tensor.to(torch.float32)

            # data type
            f.write(b'\x00')

            if len(tensor.shape) > 4:
                raise Exception('Tensors can have up to 4 dimensions')
            
            f.write(struct.pack('<B', len(tensor.shape)))
            for i in range(4):
                f.write(struct.pack('<Q', tensor.shape[i] if i < len(tensor.shape) else 0))

            data = tensor.reshape(-1).numpy(force=True)
            f.write(struct.pack('<Q', offset))
            # Assuming fp32
            offset += len(data) * 4

        for label in model_dict:
            tensor = model_dict[label]
            print(f'Writing {label} data...')
            data = tensor.to(torch.float32).reshape(-1).numpy(force=True)
            data.tofile(f)

