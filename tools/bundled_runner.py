"""Use the app's already-installed NumPy/Pandas, then workspace add-on packages.

This does not modify OS policy, unblock DLLs, or change installed packages.
"""
from pathlib import Path
import runpy
import sys
import sysconfig

sys.path.insert(0,sysconfig.get_paths()['purelib'])
import numpy
import numpy.random
import pandas

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'.deps'))
if len(sys.argv)<2:
    raise SystemExit('Usage: bundled_runner.py module [arguments]')
module=sys.argv[1]
sys.argv=[module]+sys.argv[2:]
runpy.run_module(module,run_name='__main__')
