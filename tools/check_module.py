import importlib, traceback, sys
try:
    m = importlib.import_module('app.memory_retrieve')
    print('MODULE FILE:', getattr(m, '__file__', None))
    print('HAS local_embed:', hasattr(m, 'local_embed'))
    print('PUBLIC ATTRS:', [a for a in dir(m) if not a.startswith('_')])
except Exception:
    print('--- IMPORT ERROR TRACEBACK ---')
    traceback.print_exc()
    sys.exit(1)
