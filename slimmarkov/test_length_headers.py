from length_headers import *

def test_small_numbers():
  for i in range(100000):
    value, _ = decode_length_header(encode_length_header(i))
    assert value == i

def test_large_numbers():
  for i in range(30):
    for j in range(1000):
      n = j + 2**i
      value, _ = decode_length_header(encode_length_header(n))
      assert value == n

def test_suffix_tolerance():
  suffix = "hithere"
  for i in range(10000):
    value, _ = decode_length_header(encode_length_header(i)+suffix)
    assert value == i

def test_skip_bytes():
  payload = "hello"
  data = encode_length_header(len(payload)) + payload
  value, skip = decode_length_header(data)
  retrieved = data[skip:skip+value]
  assert retrieved == payload
