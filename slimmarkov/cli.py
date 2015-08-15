import argparse
import logging
import time
import humanize

import markovify

import slimmarkov

from utils import get_ram_usage

_log_format = '%(asctime)-15s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'

def make_cli_training_module(subparsers):
  def main(args):
    with open(args.training_data, "r") as f:
      data = f.read().decode("utf-8")
    with open(args.output_model, "w") as f:
      slimmarkov.train_to_disk(markovify.Text, f, data, args.state_size)
  parser = subparsers.add_parser("train",
    help="train a markovify.Text instance from a text file to a disk model")
  parser.add_argument("training_data",
    help="filename of text file to train from")
  parser.add_argument("--output_model", required=True,
    help="output model file name")
  parser.add_argument("--state_size", type=int, default=2,
    help="model state size")
  parser.set_defaults(main=main)

def print_latencies(latencies):
  for latency in latencies:
    print "latency", latency
  for i in range(0,100+1):
    print "percentile", i, numpy.percentile(latencies, i)

def run_benchmark(f, t=60, show=True):
  logging.info("running benchmark for %d seconds", t)
  sentences = []
  latencies = []
  tzero = time.time()
  while time.time() < (tzero + t):
    t0 = time.time()
    sentences.append(f())
    t1 = time.time()
    if show:
      print sentences[-1]
    latencies.append(t1-t0)
  latencies.sort()
  logging.info("done running benchmark, collected %d entries", len(latencies))
  return latencies

def make_cli_benchmark_module(subparsers):
  def main(args):
    with open(args.model, "rb") as f:
      if args.control:
        assert args.state_size is not None
        text = markovify.Text(f.read().decode("utf-8"),
          state_size=args.state_size)
        logging.info("pure markovify model is consuming %s RAM",
          humanize.naturalsize(get_ram_usage()))
      else:
        assert args.state_size is None
        text = slimmarkov.load_from_disk(markovify.Text, f,
          cache_levels=args.cache_levels,
          cache_ratio=args.cache_ratio)
        logging.info("disk model is consuming %s RAM",
          humanize.naturalsize(get_ram_usage()))
      latencies = run_benchmark(text.make_sentence,
        t=args.seconds, show=args.show)
      print_latencies(latencies)
  parser = subparsers.add_parser("benchmark",
    help="run a disk model for a while and collect latencies")
  parser.add_argument("model",
    help="filename of disk model (or training file if control)")
  parser.add_argument("--seconds", type=float, default=60,
    help="duration to run (in seconds)")
  parser.add_argument("--cache_levels", type=int, default=3,
    help="maximum number of levels of the tree to cache")
  parser.add_argument("--cache_ratio", type=float, default=0.5,
    help="ratio of nodes to cache (selecting the heaviest)")
  parser.add_argument("--show", action="store_true",
    help="show sentences as they are generated")
  parser.add_argument("--control", action="store_true",
    help="control experiment (run plain markovify)")
  parser.add_argument("--state_size", type=int, default=None,
    help="model state size for control")
  parser.set_defaults(main=main)

def main():
  parser = argparse.ArgumentParser("slimmarkov")
  parser.add_argument("--verbose", "-v", action="count",
    help="print more logging messages")
  subparsers = parser.add_subparsers()
  make_cli_training_module(subparsers)
  make_cli_benchmark_module(subparsers)
  args = parser.parse_args()
  log_level = logging.WARNING
  if args.verbose >= 1:
    log_level = logging.INFO
  if args.verbose >= 2:
    log_level = logging.DEBUG
  logging.basicConfig(format=_log_format, level=log_level)
  logging.debug("at startup, consuming %s RAM",
    humanize.naturalsize(get_ram_usage()))
  args.main(args)

if __name__ == '__main__':
  main()
