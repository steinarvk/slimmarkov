import pbtool
import proto.slimmarkov_pb2 as pb
import StringIO

def test_roundtrip():
  f = StringIO.StringIO()
  f.write("mjau")
  message1 = pb.SymbolEntry()
  message1.data = "hello"
  message1.id = 123
  message1.frequency = 99
  message1.offset = 50
  pos1 = pbtool.write_message_at(f, message1)
  f.write("hmm" * 32)
  message2 = pb.SymbolEntry()
  message2.data = "yay" * 10001
  message2.id = 124
  message2.frequency = 95
  message2.offset = 10
  pos2 = pbtool.write_message_at(f, message2)
  f = StringIO.StringIO(f.getvalue())
  message2_ = pbtool.read_message_at(f, pos2, pb.SymbolEntry)
  message1_ = pbtool.read_message_at(f, pos1, pb.SymbolEntry)
  for a, b in ((message1, message1_), (message2,message2_)):
    assert a.data == b.data
    assert a.id == b.id
    assert a.frequency == b.frequency
    assert a.offset == b.offset
  


