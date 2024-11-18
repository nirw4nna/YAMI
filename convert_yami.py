#!/usr/bin/env python3

from enum import Enum
import argparse
import sys
from pathlib import Path
import os
import logging
from typing import List, IO, Any, Dict
import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from sentencepiece import SentencePieceProcessor
from abc import ABC, abstractmethod
import ctypes
from dataclasses import dataclass, fields
import struct


MAX_LABEL = 64
MAX_DIMS = 4

HEADER = b'\x59\x41\x4D\x49'
FILE_VERSION = 1

logger = logging.getLogger('convert_yami')


class Tokenizer(Enum):
    BPE = 0
    SP = 1

    def __str__(self) -> str:
        return f'{self.name}'
    

class Model(Enum):
    GPT2 = 0
    LLAMA = 1
    INVALID = 2

    def __str__(self) -> str:
        return f'{self.name}'

    def tokenizer(self) -> Tokenizer:
        if self == Model.GPT2:
            return Tokenizer.BPE
        elif self == Model.LLAMA:
            return Tokenizer.SP


class Dtype(Enum):
    FP32 = 0

    def __str__(self) -> str:
        return f'{self.name}'

    @staticmethod
    def get_default():
        return Dtype.FP32


DTYPE_SIZE = {
    Dtype.FP32: 4,
}

DTYPE_TO_TORCH = {
    Dtype.FP32: torch.float32,
}

HPARAMS_TYPE_MAP = {
    int: ctypes.c_uint32,
    float: ctypes.c_float
}

@dataclass
class Hparams(ABC):
    
    def to_raw(self) -> bytearray:
        struct_fields = [(field.name, HPARAMS_TYPE_MAP[field.type]) for field in fields(self)]
        struct_class = type('YamiHparams', (ctypes.Structure, ), {'_fields_': struct_fields})
        the_struct = struct_class(*[getattr(self, field) for field, _ in struct_fields])
        return bytearray(the_struct)


@dataclass
class GPT2Hparams(Hparams):
   n_layers: int
   n_heads: int
   emb_size: int
   ctx_size: int
   vocab_size: int
   norm_eps: float


@dataclass
class LlamaHparams(Hparams):
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int 
    norm_eps: float
    ctx_size: int


"""
Format specification for the YAMI file:
    Both the tokenizer and weights are exported to the same file to make running models more convenient.
    For now, the ordering is assumed to be always little-endian.


    The structure of the file is the following: <header> <file version> <tokenizer> <model metadata> <tensors metadata> <tensors>

    Header (4 bytes): 0x59 0x41 0x4D 0x49
    
    File version (1 byte): 1

    Tokenizer:
        - tokenizer type (1 byte: 0=BPE, 1=SP)
        - if SP:
            -- vocab size (4 bytes)
            -- foreach vocab:
                --- size (2 bytes)
                --- vocab (size bytes, UTF-8)
                --- score (4 bytes, float)
        - if BPE:
            -- encoder size (4 bytes)
            -- foreach token:
                --- size (2 bytes)
                --- token (size bytes, UTF-8)
            -- bpe merges size / vocab (4 bytes)
            -- foreach bpe pair:
                --- size (2 bytes)
                --- bpe pair (size bytes, two UTF-8 strings separated by whitespace)

    Model:
    - model type (1 byte: 0=GPT-2, 1=LLaMA)
    - hparams size (2 bytes)
    - hparams
    - number of tensors (2 bytes)
    - foreach tensor:
        -- label (64 bytes, ASCII)
        -- encoding (1 byte: 0=FP32)
        -- number of dimensions (1 byte)
        -- dimensions (8 bytes per dimension, always 4 dimensions)
        -- data offset (8 bytes)
    - foreach tensor:
        -- data (size bytes in row-major order, encoded as specified)
"""
class ModelExporter(ABC):
    def __init__(self, out_dtype: Dtype, ctx_size: int):
        self.out_dtype = out_dtype
        self.ctx_size = ctx_size

    def export(self, out_file: str, to_ignore: List[str] = None, to_transpose: List[str] = None):
        logger.info(f'Exporting {self.model_type} model to "{out_file}"')
        
        self._load_model_dict()
        
        hparams = self._hparams()

        tensors_to_remove = set()
        for label in self.model_dict:
            if to_ignore and any([label.endswith(ti) for ti in to_ignore]):
                tensors_to_remove.add(label)
                logger.debug(f'Tensor "{label}" will be ignored')
            
            if to_transpose and any([label.endswith(ti) for ti in to_transpose]):
                logger.debug(f'Tensor "{label}" with shape {self.model_dict[label].shape} will be transposed')
                self.model_dict[label] = self.model_dict[label].t()

        self.model_dict = {k: v for k, v in self.model_dict.items() if k not in tensors_to_remove}

        logger.debug(f'Removed {len(tensors_to_remove)} tensors from model_dict')

        with open(out_file, mode='wb') as f:
            # Header and version number
            f.write(HEADER)
            f.write(struct.pack('<B', FILE_VERSION))
            
            # Tokenizer
            self._export_tokenizer(f)

            # Model type
            f.write(struct.pack('<B', self.model_type.value))
            
            # Hparams
            raw_hparams = hparams.to_raw()
            f.write(struct.pack('<H', len(raw_hparams)))
            f.write(raw_hparams)

            # Model
            self._export_model(f)

        logger.info(f'Correctly exported {self.model_type} model to "{out_file}"')

    @abstractmethod
    def _load_model_dict(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _hparams(self) -> Hparams:
        pass

    @abstractmethod
    def _export_tokenizer(self, file_handle: IO):
        pass

    def _export_model(self, file_handle: IO):
        offset = 0

        yami_dtype = self.out_dtype if self.out_dtype in DTYPE_TO_TORCH else Dtype.get_default()
        expected_type = DTYPE_TO_TORCH[yami_dtype]
        
        logger.debug(f'Tensors will be exported as {expected_type} [{yami_dtype}]')

        # Number of tensors that will be exported
        file_handle.write(struct.pack('<H', len(self.model_dict)))

        # Export metadata first
        for label in self.model_dict:
            tensor = self.model_dict[label]
            logger.info(f'Found tensor "{label}" | {tensor.shape} | {tensor.dtype}')

            label_len = len(label)
            if label_len > MAX_LABEL:
                logger.warn(f'Label "{label}" is too long and will be clipped')

            file_handle.write(bytes(label[:MAX_LABEL], encoding='ascii') + (b'\x00' * (64 - label_len) if label_len <= MAX_LABEL else b''))

            # Check if the tensor has the correct type or if we have to convert it
            if tensor.dtype != expected_type:
                logger.debug(f'Dtype missmatch - "{label}" {tensor.dtype} ==> {expected_type}')
                tensor = tensor.to(expected_type)

            file_handle.write(struct.pack('<B', yami_dtype.value))

            if len(tensor.shape) > MAX_DIMS:
                logger.error(f'"{label}" has more than {MAX_DIMS} dimensions')
                sys.exit(1)

            file_handle.write(struct.pack('<B', len(tensor.shape)))
            for i in range(MAX_DIMS):
                file_handle.write(struct.pack('<Q', tensor.shape[i] if i < len(tensor.shape) else 0))

            data = tensor.reshape(-1).numpy(force=True)
            file_handle.write(struct.pack('<Q', offset))
            
            logger.debug(f'"{label}" offset: {offset}')

            offset += len(data) * DTYPE_SIZE[yami_dtype]
        
        # Write the actual data
        for label in self.model_dict:
            tensor = self.model_dict[label]
            data = tensor.to(expected_type).reshape(-1).numpy(force=True)
            data.tofile(file_handle)
            logger.debug(f'"{label}" - wrote {len(data)} bytes')


class LLaMAExporter(ModelExporter):
    def __init__(self, weights: str, tokenizer: str, params: str,
                 out_dtype: Dtype, ctx_size: int):
        super().__init__(out_dtype, ctx_size)
        self.model_type = Model.LLAMA
        self.weights = weights
        self.tokenizer = tokenizer
        self.params = params

    def _load_model_dict(self):
        self.model_dict = torch.load(self.weights)
        logger.debug(f'Loaded PyTorch model_dict from "{self.weights}" - found {len(self.model_dict)} tesors')

    def _hparams(self) -> Hparams:
        with open(self.params, 'rt') as f:
            params = json.load(f)

            logger.debug(f'Loaded parameters from "{self.params}"')

            hparams = LlamaHparams(
                dim=params['dim'],
                n_layers=params['n_layers'],
                n_heads=params['n_heads'],
                norm_eps=params['norm_eps'],
                vocab_size=self.model_dict['tok_embeddings.weight'].shape[0],
                ctx_size=self.ctx_size
            )

            logger.debug(hparams)
            return hparams

    def _export_tokenizer(self, file_handle: IO):
        logger.debug(f'Exporting {Tokenizer.SP} tokenizer')
        file_handle.write(struct.pack('<B', Tokenizer.SP.value))

        sp = SentencePieceProcessor(model_file=self.tokenizer)
        n_words = sp.vocab_size()
        bos_id = sp.bos_id()
        eos_id = sp.eos_id()
        file_handle.write(struct.pack('<I', n_words))
        for i in range(n_words):
            token = sp.id_to_piece(i)
            score = sp.get_score(i)
            
            if i == bos_id:
                token = '<s>'
            elif i == eos_id:
                token = '</s>'

            token = token.replace('▁', ' ')
            raw_token = bytes(token, encoding='utf-8')
            file_handle.write(struct.pack('<H', len(raw_token)))
            file_handle.write(raw_token)
            file_handle.write(struct.pack('<f', score))

        logger.debug(f'{Tokenizer.SP} tokenizer exported correctly')


class GPT2Exporter(ModelExporter):
    def __init__(self, out_dtype: Dtype, ctx_size: int):
        super().__init__(out_dtype, ctx_size)
        self.model_type = Model.GPT2

    def _load_model_dict(self):
        # Note: for some reason HF doesn't like if there is another file named 'gpt2' so,
        # before converting a GPT2 model, remember to delete that file
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model_dict = model.state_dict()
        
        # The tensors loaded from HF must be transposed
        # for label in self.model_dict.keys():
        #     if label.endswith('.weight'):
        #         self.model_dict[label] = self.model_dict[label].t()

        logger.debug(f'Loaded GPT2 model from Hugging Face - found {len(self.model_dict)} tesors')

    def _hparams(self) -> Hparams:
        # gpt2-small
        config = GPT2Config()
        return GPT2Hparams(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            emb_size=config.n_embd,
            ctx_size=self.ctx_size,
            vocab_size=config.vocab_size,
            norm_eps=config.layer_norm_epsilon,
        )

    def _export_tokenizer(self, file_handle: IO):
        logger.debug(f'Exporting {Tokenizer.BPE} tokenizer')
        file_handle.write(struct.pack('<B', Tokenizer.BPE.value))
        
        bpe = GPT2Tokenizer.from_pretrained('gpt2')
        encoder = bpe.encoder
        file_handle.write(struct.pack('<I', len(encoder)))
        
        for k in encoder:
            raw = bytes(k, encoding='utf-8')
            file_handle.write(struct.pack('<H', len(raw)))
            file_handle.write(raw)
        
        bpe_ranks = bpe.bpe_ranks
        file_handle.write(struct.pack('<I', len(bpe_ranks)))
        for k in bpe_ranks:
            bpe_pair = ' '.join(k)
            raw = bytes(bpe_pair, encoding='utf-8')
            file_handle.write(struct.pack('<H', len(raw)))
            file_handle.write(raw)

        logger.debug(f'{Tokenizer.BPE} tokenizer exported correctly')


def export_model(path: Path, dtype: Dtype, ctx_size: int, out_file: Path):
    if not os.path.exists(path):
        logger.error(f'Path "{path}" not valid')
        sys.exit(1)

    model_folder = os.path.basename(path)

    to_ignore = []
    to_transpose = []
    if 'llama' in model_folder:
        # For now, if the model is LLaMA then folder must contain:
        #   - tokenizer.model
        #   - params.json
        #   - *.pth
        checkpoint_path = list(Path(path).glob('consolidated.*.pth'))[0]
        if not os.path.exists(checkpoint_path):
            logger.error(f'PyTorch checkpoints not found in "{path}"')
            sys.exit(1)

        params_path = os.path.join(path, 'params.json')
        if not os.path.exists(params_path):
            logger.error(f'Params not found in "{path}"')
            sys.exit(1)

        tokenizer_path = os.path.join(path, 'tokenizer.model')
        if not os.path.exists(params_path):
            logger.error(f'Tokenizer not found in "{path}"')
            sys.exit(1)


        exporter = LLaMAExporter(checkpoint_path, tokenizer_path, 
                                 params_path, dtype, ctx_size)
        to_ignore.append('rope.freqs')
    elif 'gpt2' in model_folder:
        # In this case we don't do any check, just load the basic gpt2-small model from HF
        exporter = GPT2Exporter(dtype, ctx_size)
        to_transpose.extend(['attn.c_attn.weight', 'attn.c_proj.weight',
                             'mlp.c_fc.weight', 'mlp.c_proj.weight'])
    else:
        logger.error(f'Unknown model "{model_folder}"')
        sys.exit(1)

    out_file = os.path.join(path, f'yami_{model_folder}_{str(dtype).lower()}.bin') if out_file is None else out_file
    exporter.export(out_file, to_ignore=to_ignore, to_transpose=to_transpose)


if __name__ == '__main__':
    dtype_choices = [dtype.name.lower() for dtype in Dtype]
    default_dtype = Dtype.get_default().name.lower()
    cli_parser = argparse.ArgumentParser(
        description='Conversion utility for YAMI',
    )
    cli_parser.add_argument('path', type=Path, help='Path to the folder containing the weights of the model')
    cli_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    cli_parser.add_argument('--out-type', choices=dtype_choices,
                            default=default_dtype,
                            help=f'Data type to be used to export the model weights (default={default_dtype})')
    cli_parser.add_argument('-o', '--output', type=Path, help='Path to the output file', default=None)
    cli_parser.add_argument('--ctx', type=int, default=512,
                            help='Maximum context size for this model (default=512)')

    args = cli_parser.parse_args()

    # TODO: better formatting
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    out_dtype = Dtype[args.out_type.upper()]

    export_model(args.path, out_dtype, args.ctx, args.output)
