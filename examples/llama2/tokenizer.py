# Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the MIT license
# (https://opensource.org/license/mit).

from sentencepiece import SentencePieceProcessor
from typing import List


class LlamaTokenizer:
    def __init__(self, tokenizer_model: str):
        self.sp = SentencePieceProcessor(model_file=tokenizer_model)
        self.n_words = self.sp.vocab_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        print(f'Loaded tokenizer from {tokenizer_model}')

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        idx = self.sp.encode(s)
        if bos:
            idx = [self.bos_id] + idx
        
        if eos:
            idx = idx + [self.eos_id]
        
        return idx

    def decode(self, idx: List[int]) -> str:
        return self.sp.decode(idx)
