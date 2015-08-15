import length_headers

_buffer_guess = 4096

def write_message_at(f, message):
  pos = f.tell()
  data = message.SerializeToString()
  header = length_headers.encode_length_header(len(data))
  f.write(header)
  f.write(data)
  return pos

def read_message_at(f, pos, message_class, seek_after=False):
  f.seek(pos)
  message = message_class()
  data = f.read(_buffer_guess)
  datalen, skip = length_headers.decode_length_header(data)
  if len(data) < (datalen + skip):
    data += f.read(datalen + skip - len(data))
  payload = data[skip:skip+datalen]
  message.ParseFromString(payload)
  if seek_after:
    f.seek(pos + datalen + skip)
  return message

