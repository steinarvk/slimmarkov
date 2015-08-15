import collections
import numpy
import zlib
import humanize
import datetime
import time
import logging
import hashlib
import random
import os
import struct
import markovify

from .pbtool import (read_message_at, write_message_at)
from .proto.slimmarkov_pb2 import (MarkovTree, Header, SymbolTable)
from .utils import (bisect_right_with_key, bisect_left_with_key,
                    select_top_ratio, get_ram_usage)


__author__ = "Steinar V. Kaldager"
__version__ = "0.1.0"


BEGIN = "symbol:BEGIN"
END = "symbol:END"

_magic_string = "MarkovModel"

_prefix = "w:"

def word(s):
  return _prefix + s

def unword(w):
  assert w.startswith(_prefix)
  return w[len(_prefix):]

CachedBranch = collections.namedtuple("CachedBranch",
  "symbol_id cumulative_weight next_node_offset")
CachedNode = collections.namedtuple("CachedNode",
  "symbol_id total_weight branches")

class MarkovifyInterface(object):
  def __init__(self, model):
    self.model = model

  def walk(self, initial_state=None):
    n = self.model.header.state_size
    state = list(initial_state or (BEGIN,) * n)
    assert len(state) == n
    while state[-1] != END:
      old_state = state[-n:]
      state.append(self.model.lookup(state[-n:]))
    return map(unword, state[n:-1])

def translate_markovify_word(w):
  if w == "___BEGIN__":
    return BEGIN
  if w == "___END__":
    return END
  return word(w)

class SymbolTableEntry(object):
  def __init__(self, data, index, frequency, offset=None):
    self.data = data
    self.index = index
    self.frequency = frequency
    self.offset = offset

  def __dict__(self):
    return {
      "data": self.data,
      "index": self.index,
      "frequency": self.frequency,
      "offset": self.offset,
    }

def build_symbol_table(markovify_chain):
  ctr = collections.Counter()
  for nm1gram, leaf_items in markovify_chain.model.items():
    for leaf, weight in leaf_items.items():
      for chain_word in (nm1gram + (leaf,)):
        ctr[translate_markovify_word(chain_word)] += weight
  rv = []
  for index, (word, weight) in enumerate(ctr.most_common()):
    rv.append(SymbolTableEntry(word, index, weight))
  return rv

class MarkovNode(object):
  def __init__(self, symbol_id, weight):
    self.symbol_id = symbol_id
    self.weight = weight
    self.offset = None
    self.children = []

  def write(self, out):
    self.children.sort(key=lambda c: c.symbol_id)
    cum_weight = 0
    node_pb = MarkovTree(
      symbol_id=self.symbol_id,
      total_weight=self.weight,
    )
    for child in self.children:
      branch_pb = node_pb.branches.add()
      cum_weight += child.weight
      branch_pb.cumulative_weight = cum_weight
      branch_pb.symbol_id = child.symbol_id
      if child.children:
        assert child.offset is not None
        branch_pb.next_node_offset = child.offset
    self.offset = write_message_at(out, node_pb)

def write_symbol_table(symbols, out):
  table_pb = pb.SymbolTable()
  for symbol in symbols:
    symbol_pb = table_pb.entry.add()
    symbol_pb.data = symbol.data
    symbol_pb.id = symbol.index
    symbol_pb.frequency = symbol.frequency
    if symbol.offset is not None:
      symbol_pb.offset = symbol.offset
  logging.info("writing %d entries to symbol table", len(table_pb.entry))
  return write_message_at(out, table_pb)

class Model(object):
  def __init__(self, header, symbols, handle, rand):
    self.header = header
    self.symbols = symbols
    self.symbols_by_name = {s.data: s for s in symbols}
    self.handle = handle
    self.rand = rand
    self.cache = {}

  def cache_node(self, pos):
    node = self.read_node(pos)
    cached_branches = [CachedBranch(
      symbol_id=b.symbol_id,
      cumulative_weight=b.cumulative_weight,
      next_node_offset=b.next_node_offset,
    ) for b in node.branches]
    rv = self.cache[pos] = CachedNode(
      symbol_id=node.symbol_id,
      total_weight=node.total_weight,
      branches=cached_branches,
    )
    return rv

  def cache_nodes_below(self, pos, n, ratio):
    node = self.cache_node(pos)
    if n <= 0:
      return
    branches = []
    last_weight = 0
    for branch in node.branches:
      weight = branch.cumulative_weight - last_weight
      branches.append((weight, branch))
    best_branches = select_top_ratio(branches, ratio)
    logging.debug("selected %d out of %d branches for caching",
      len(best_branches), len(branches))
    for _, branch in best_branches:
      if not branch.next_node_offset:
        continue
      self.cache_nodes_below(branch.next_node_offset, n-1, ratio)

  def enable_caching(self, n, ratio):
    best_symbols = select_top_ratio(self.symbols, ratio,
      lambda s: s.frequency)
    logging.info("selected %d out of %d symbols for caching",
      len(best_symbols), len(self.symbols))
    for sym in best_symbols:
      if not sym.offset:
        continue
      self.cache_nodes_below(sym.offset, n-1, ratio)

  def read_node(self, pos):
    try:
      return self.cache[pos]
    except KeyError:
      return read_message_at(self.handle, pos, MarkovTree)

  def draw_from_node(self, node):
    w = self.rand.randint(0, node.total_weight - 1)
    index = bisect_right_with_key(node.branches, w,
      lambda b: b.cumulative_weight)
    branch = node.branches[index]
    assert w < branch.cumulative_weight
    assert (index == 0) or node.branches[index-1].cumulative_weight <= w
    return branch.symbol_id

  def lookup_from_node(self, ngram, node):
    if not ngram:
      return self.draw_from_node(node)
    index = bisect_left_with_key(node.branches, ngram[0],
      lambda b: b.symbol_id)
    branch = node.branches[index]
    assert branch.symbol_id == ngram[0]
    next_node = self.read_node(branch.next_node_offset)
    return self.lookup_from_node(ngram[1:], next_node)

  def lookup(self, ngram):
    idngram = map(lambda x: self.symbols_by_name[x].index, ngram)
    sym = self.symbols[idngram[0]]
    node = self.read_node(sym.offset)
    idrv = self.lookup_from_node(idngram[1:], node)
    return self.symbols[idrv].data

def read_model_training_data(f):
  magic = f.read(len(_magic_string))
  if magic != _magic_string:
    raise ValueError("wrong file header")
  header = read_message_at(f, f.tell(), Header,
    seek_after=True)
  data = zlib.decompress(f.read(header.data_length))
  if hashlib.sha1(data).hexdigest() != header.data_sha1:
    raise ValueError("invalid data section")
  return data.decode("utf-8")

def read_model(f, size):
  magic = f.read(len(_magic_string))
  if magic != _magic_string:
    raise ValueError("wrong file header")
  header = read_message_at(f, f.tell(), Header)
  logging.info("read header:")
  for line in str(header).splitlines():
    logging.info("  " + line)
  footer_size = 8
  f.seek(size - footer_size)
  st_offset = struct.unpack("<Q", f.read(footer_size))[0]
  logging.info("reading symbol table..")
  symbol_table = read_message_at(f, st_offset, SymbolTable)
  symbols = []
  logging.info("%d symbols in symbol table", len(symbol_table.entry))
  for entry in symbol_table.entry:
    symbols.append(SymbolTableEntry(
      entry.data, entry.id, entry.frequency, entry.offset))
  return Model(header, symbols, f, random)
    
def build_disk_model(markovify_chain, out,
    data=None, model_class=None):
  out.write(_magic_string)
  header = Header(
    version = __version__,
    state_size = markovify_chain.state_size,
    timestamp_utc = datetime.datetime.utcnow().isoformat(),
  )
  if model_class:
    try:
      header.model_class = model_class.__module__ + "." + model_class.__name__
    except AttributeError as e:
      logging.warning("no valid name for: %s", repr(model_class))
      logging.exception(e)
  if data:
    data = data.encode("utf-8")
    header.data_sha1 = hashlib.sha1(data).hexdigest()
    compressed_data = zlib.compress(data)
    header.data_length = len(compressed_data)
  write_message_at(out, header)
  if data:
    out.write(compressed_data)
  symbols = build_symbol_table(markovify_chain)
  symbols_by_name = {s.data: s for s in symbols}
  symbols_by_id = {s.index: s for s in symbols}
  node_chunks = {}
  next_parentage = collections.defaultdict(lambda: [])
  def chain_word_id(w):
    return symbols_by_name[translate_markovify_word(w)].index
  for nm1gram, leaf_items in markovify_chain.model.items():
    for leaf, weight in leaf_items.items():
      ngram = map(chain_word_id, (nm1gram + (leaf,)))
      leaf_node = MarkovNode(chain_word_id(leaf), weight)
      next_parentage[tuple(map(chain_word_id,nm1gram))].append(leaf_node)
  level_no = 0
  while next_parentage:
    parentage = next_parentage
    next_parentage = collections.defaultdict(lambda: [])
    pos0 = out.tell()
    for key, child_nodes in parentage.items():
      node = MarkovNode(key[-1], sum(n.weight for n in child_nodes))
      node.children.extend(child_nodes)
      node.write(out)
      if len(key) > 1:
        next_parentage[key[:-1]].append(node)
      else:
        symbols_by_id[key[0]].offset = node.offset
    pos1 = out.tell()
    size = pos1 - pos0
    logging.debug("on level %d, %d nodes (%s)", level_no,
      len(parentage), humanize.naturalsize(size))
    level_no += 1
  symbol_table_offset = write_symbol_table(symbols, out)
  out.write(struct.pack("<Q", symbol_table_offset))

def train_to_disk(markovify_class, f, data, state_size):
  """Train a Markov model, convert it to a disk-readable format, and save it.

  This function uses markovify machinery (or markovify-compatible machinery)
  to train the model in the first place. This has the advantage that lots
  of finicky natural-language code is already written for it, e.g.
  part-of-speech tagging with nltk and retries to avoid regenerating the
  source material.

  Args:
    - markovify_class: class to use, e.g. markovify.text.Text
    - f: file handle of the output file to which the model will be written
    - data: input data, with which the provided class will be trained,
            and which will also be saved in the model file
    - state_size: how many tokens to keep in order to predict the next token.
  """
  logging.info("training %d-gram model from %s of data..",
    state_size, humanize.naturalsize(len(data)))
  text = markovify_class(data, state_size)
  logging.info("model trained in memory, building to disk..")
  logging.info("consuming %s RAM", humanize.naturalsize(get_ram_usage()))
  build_disk_model(text.chain, f, data=data, model_class=markovify_class)
  f.flush()
  size = os.fstat(f.fileno()).st_size
  logging.info("model built and written to disk (%s)",
    humanize.naturalsize(size))

def load_from_disk(markovify_class, f, cache_levels, cache_ratio):
  """Load a Markov model from a saved file on disk and wrap it in a class.

  Args:
    - markovify_class: markovify-compatible class (or callable),
                       e.g. markovify.text.Text
    - f: file handle of model file, which needs to be kept open while
         using the markov model
    - cache_levels: maximum number of levels of the tree to cache
    - cache_ratio: ratio of nodes that are cached at each level (e.g.
                   the 20% most frequent nodes at each eligible level).
  """
  f.seek(0)
  logging.info("reading training data..")
  data = read_model_training_data(f)
  logging.info("read %s of training data", humanize.naturalsize(len(data)))
  size = os.fstat(f.fileno()).st_size
  f.seek(0)
  logging.info("reading model of %s..", humanize.naturalsize(size))
  model = read_model(f, size)
  logging.info("%d-gram model read, total RAM consumption %s",
    model.header.state_size, humanize.naturalsize(get_ram_usage()))
  if cache_levels > 0 and cache_ratio > 0:
    logging.info("caching up to %d levels of the %0.2lf%% most frequent nodes",
      cache_levels, cache_ratio * 100)
    model.enable_caching(cache_levels, cache_ratio)
    logging.info("caching enabled, total RAM consumption %s",
      humanize.naturalsize(get_ram_usage()))
  chain_like = MarkovifyInterface(model)
  return markovify_class(
    data,
    state_size=model.header.state_size,
    chain=chain_like,
  )
