import resource

def bisect_right_with_key(xs, xval, key, low=0, high=None):
  if high is None:
    high = len(xs)
  while low < high:
    mid = (low + high) // 2
    if xval < key(xs[mid]):
      high = mid
    else:
      low = mid + 1
  return low

def bisect_left_with_key(xs, xval, key, low=0, high=None):
  if high is None:
    high = len(xs)
  while low < high:
    mid = (low + high) // 2
    if key(xs[mid]) < xval:
      low = mid + 1
    else:
      high = mid
  return low

def select_top_ratio(xs, ratio, key=None):
  assert 0 <= ratio <= 1.0
  xs = list(xs)
  xs.sort(reverse=True, key=key)
  n = int(len(xs)*float(ratio))
  return xs[:n]

def get_ram_usage():
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

