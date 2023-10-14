import torch
import struct
import ctypes
import os


class YamiHparams(ctypes.Structure):
    pass

"""
YAMI format:
Binary file, little endian.

- Magic (3 bytes) --> 0x1A 0xAA 0x1F
- Size of hyperparameters (2 bytes)
- Hyperparameters
- Number of tensors (2 bytes)

foreach tensor:
- Label size (2 bytes)
- Label
- Encoding (1 byte) --> for now just 0 for fp32
- Number of dimensions (1 byte)
- Dimensions (always 4 dimensions, 4 bytes per dimension)
- Number of elements (8 bytes)
- Data --> flat array (row-major), encoded as specified
"""

def _encode_tensor(label, tensor, fout):
    fout.write(struct.pack('<H', len(label)+1))
    fout.write(bytes(label, encoding='ascii') + b'\x00')

    if not tensor.dtype == torch.float32:
        raise Exception(f'Unknow type {tensor.dtype}')

    # tensor data type
    fout.write(b'\x00')

    if len(tensor.shape) > 4:
        raise Exception('Tensors can have up to 4 dimensions')
    
    fout.write(struct.pack('<B', len(tensor.shape)))
    for i in range(4):
        fout.write(struct.pack('<I', tensor.shape[i] if i < len(tensor.shape) else 0))


    data = tensor.view(-1).numpy(force=True)
    fout.write(struct.pack('<Q', len(data)))
    
    data.tofile(fout)


def export_model(model_dict, hparams, dest):
    print(f'Exporting model with {len(model_dict)} tensors to "{dest}"')
    
    if os.path.isfile(dest):
        print(f'"{dest}" will be overwritten')
        os.remove(dest)

    with open(dest, 'xb') as out_file:
        out_file.write(b'\x1A\xAA\x1F') # header
        hp_b = bytearray(hparams)
        out_file.write(struct.pack('<H', len(hp_b))) # size of hparams
        out_file.write(hp_b)
        out_file.write(struct.pack('<H', len(model_dict))) # number of tensors

        for label in model_dict:
            t = model_dict[label]
            print(f'Found new tensor: [ {label} {t.shape} {t.dtype} ]')
            _encode_tensor(label, t, out_file)