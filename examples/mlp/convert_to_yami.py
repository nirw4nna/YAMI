import torch
import struct

"""
YAMI format:
Binary file, little endian.

- Magic (3 bytes) --> 0x1A 0xAA 0x1F
- Total number of tensors (2 bytes)

foreach tensor:
- Label (64 bytes, fixed)
- Encoding (1 byte) --> for now just 0 for fp32
- Number of dimensions (1 byte)
- Dimensions (always 4 dimensions, 4 bytes per dimension)
- Number of elements (4 bytes)
- Data --> flat array (row-major), encoded as specified
"""

def encode_tensor(label, tensor):
    buff = b''
    if len(label) >= 64:
        raise Exception(f'Label {label} has too many characters')
    
    buff += bytes(label, encoding='ascii')
    buff += (b'\0' * (64 - len(label)))

    if not tensor.dtype == torch.float32:
        raise Exception(f'Unknow type {tensor.dtype}')
    
    buff += b'\0'

    if len(tensor.shape) > 4:
        raise Exception('Tensors can have up to 4 dimensions')
    
    buff += struct.pack('<B', len(tensor.shape))
    for i in range(4):
        buff += struct.pack('<I', tensor.shape[i] if i < len(tensor.shape) else 0)


    data = tensor.view(-1).tolist()
    buff += struct.pack('<I', len(data))
    
    for d in data:
        buff += struct.pack('<f', d)
    
    return buff


def from_model_dict(model_path, dest):
    print(f'Loading {model_path}')
    model_dict = torch.load(model_path)
    print(f'Found {len(model_dict)} tensors')
    
    out_file = open(dest, 'xb')
    out_file.write(b'\x1A\xAA\x1F') # header
    out_file.write(struct.pack('<H', len(model_dict))) # number of tensors
    
    for label in model_dict:
        t = model_dict[label]
        print(f'Tensor {label} {t.shape}, {t.dtype}')
        out_file.write(encode_tensor(label, t))

    out_file.close()

from_model_dict('simple_mlp.ptrch', 'model.yami')