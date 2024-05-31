import regex as re
import os
import requests
import json
from transformers import GPT2Tokenizer

def bytes_to_unicode():
    """
    Return a mapping between every possible byte and a unicode char.
    The bytes that can't be printed are simply shifted by 256.
    """
    raw_b = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    pretty_b = raw_b[:] # pretty_b is a deep copy of raw_b!

    n = 0
    for b in range(2**8):
        if b not in raw_b:
            raw_b.append(b)
            pretty_b.append(2**8+n)
            n += 1

    return dict(zip(raw_b, [chr(b) for b in pretty_b]))


def get_pairs(word):
    """
    Return all the bigrams in word as a set of tuples.
    """
    pairs = set()
    prev = word[0]
    for c in word[1:]:
        pairs.add((prev, c))
        prev = c
    
    return pairs


def get_file_if_missing(local_file, remote_path):
    if not os.path.isfile(local_file):
        # download from OpenAI
        resp = requests.get(remote_path)
        with open(local_file, 'wb') as f:
            f.write(resp.content)


class Encoder:
    """
    The encoder is a dict that can be loaded from 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'.
    The bpe_merges can be loaded from 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    """
    def __init__(self, encoder, bpe_merges):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.encoder = encoder
        self.decoder = {v:k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        """
        This pattern is looking for common contractions (like "you're", "it's"...) and making those into separate tokens
        then, separate everything else into consecutive chunks of letters, numbers, symbols (non-letter-numbers), whitespaces.
        Taken from https://github.com/openai/gpt-2/blob/master/src/encoder.py
        """
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


    @classmethod
    def from_pretrained(cls):
        work_dir = os.getcwd()
        encoder_file = f'{work_dir}/encoder.json'
        bpe_file = f'{work_dir}/vocab.bpe'

        get_file_if_missing(encoder_file, 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json')
        get_file_if_missing(bpe_file, 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe')

        with open(encoder_file, 'r') as f:
            encoder = json.load(f)
        assert len(encoder) == 50257

        with open(bpe_file, mode='r', encoding='utf-8') as f:
            bpe = f.read()
        bpe_lines = bpe.split('\n')
        bpe_merges = [tuple(bpe_line.split()) for bpe_line in bpe_lines[1:-1]]
        return Encoder(encoder, bpe_merges)


    def merge(self, token):
        # TODO: cache?
        word = tuple(token)
        bigrams = get_pairs(word)

        if not bigrams:
            return token

        while True:
            # Find the next, lowest rank bigram that can be merged
            next_bigram = min(bigrams, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if next_bigram not in self.bpe_ranks:
                # bpe_ranks.get returned 'inf' aka we are done here
                break
            
            first, second = next_bigram

            # Look for all the occurences of (first, second) in the list of current words
            # and merge them into one token first_second.
            new_word = []
            i = 0
            while i < len(word):

                # find the first occurence of first in word
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    # there are no occurences of first in word, done
                    new_word.extend(word[i:])
                    break
                    
                # if first is also followed by second, merge them into one
                if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(first) #TODO: check
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                bigrams = get_pairs(word)

        return ' '.join(word)


    def encode(self, text):
        """
        Given a text string return a it's BPE encoding as a list of integers.
        """
        bpe_idx = []
        
        tokens = re.findall(self.pattern, text)
        for token in tokens:
            token_bytes = token.encode(encoding='utf-8')
            # get the printable representation of each byte in token
            translated_token = ''.join(self.byte_encoder[b] for b in token_bytes)
            # perform the merges
            merged_token = self.merge(translated_token).split(' ')
            # translate all the merged token to integers and append them to our output list
            bpe_idx.extend([self.encoder[bpe_token] for bpe_token in merged_token])
        
        return bpe_idx


    def decode(self, bpe_idx):
        """
        Given a list of integers corresponding to the BPE encoding, return the corresponding string
        """
        merged_token = [self.decoder[b] for b in bpe_idx]
        token_flat = ''.join(merged_token)
        token_bytes = bytearray([self.byte_decoder[c] for c in token_flat])
        return token_bytes.decode(encoding='utf-8', errors='replace')


if __name__ == '__main__':
    # run some tests
    # Note: with the string "~tok ` ' " the HF tokenizer seems to be bugged...
    tests = ["Hello, it's me, Mario!", "Hi!", "", " #èò", "~tok `' "]
    tokenizer = Encoder.from_pretrained()
    hf_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print(hf_tokenizer.encoder)
    print(hf_tokenizer.bpe_ranks)
    # for test_str in tests:
    #     print(f'Testing \"{test_str}\"...', end='')
    #     idx = tokenizer.encode(test_str)
    #     hf_idx = hf_tokenizer(test_str)['input_ids']
    #     assert idx == hf_idx

    #     decoded = tokenizer.decode(idx)
    #     hf_decoded = hf_tokenizer.decode(hf_idx)
    #     assert hf_decoded == test_str and decoded == hf_decoded

    #     print('OK!')