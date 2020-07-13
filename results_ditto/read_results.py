import os

from tensorflow.python.summary.summary_iterator import summary_iterator
from glob import glob

rank_field = 'f1'
fields = ['t_acc', 't_f1']
runs = glob("*/")
results = {}

def get_latest(events):
  best = 0.0
  res = events[0]
  for event_fn in events:
    tm = os.path.getmtime(event_fn)
    if tm > best:
      best = tm
      res = event_fn
  return res


for run in runs:
  try:
    target_events = glob('%s%s/*' % (run, rank_field))
    values = []

    event_fn = get_latest(target_events)
    for e in summary_iterator(event_fn):
      for v in e.summary.value:
        values.append(v.simple_value)

    for field in fields:
      target_events = glob('%s%s/*' % (run, field))
      o_values = []
      event_fn = get_latest(target_events)
      for e in summary_iterator(event_fn):
        for v in e.summary.value:
          o_values.append(v.simple_value)
      max_val = 0.0
      result = 0.0
      for v, ov in zip(values[5:], o_values[5:]):
        if v > max_val:
          max_val = v
          result = ov

      print(run, field, result)

      if run not in results:
        results[run] = {}
      results[run][field] = result
  except:
    pass

