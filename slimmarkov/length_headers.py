import struct

def encode_length_header(l):
  assert l >= 0
  header_length_bits = 6
  if l < 2**(header_length_bits):
    return struct.pack("<B", l)
  if l < 2**(header_length_bits+8):
    payload_fmt = "B"
    payload_val = l & 0xff
    header_data = (l >> 8)
    header_bits = 1 << header_length_bits
  elif l < 2**(header_length_bits+16):
    payload_fmt = "H"
    payload_val = l & 0xffff
    header_data = (l >> 16)
    header_bits = 2 << header_length_bits
  elif l < 2**32:
    payload_fmt = "I"
    payload_val = l
    header_data = 0 # extended header!
    header_bits = 3 << header_length_bits
  else:
    raise ValueError("too big: {}".format(l))
  assert (header_data & 0xc0) == 0
  return struct.pack("<B" + payload_fmt, header_data | header_bits, payload_val)

def decode_length_header(s):
  header_byte = struct.unpack("<B", s[0])[0]
  header_type = header_byte >> 6
  header_data = header_byte & 0x3f
  if header_type == 3:
    if header_data != 0:
      raise ValueError("unknown extended header flag")
    all_data = struct.unpack("<I", s[1:1+4])[0]
    return all_data, 5
  if header_type == 1:
    more_data = struct.unpack("<B", s[1])[0]
    return more_data | (header_data << 8), 2
  if header_type == 2:
    more_data = struct.unpack("<H", s[1:1+2])[0]
    return more_data | (header_data << 16), 3
  assert header_type == 0
  return header_data, 1

