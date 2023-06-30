import sys
from pprint import pprint

pprint(sys.path)

sys.path.append('C:\Program Files\Python36\envs\AutoScript')
sys.path.append('C:\Program Files\Python36\envs\AutoScript\Lib\site-packages')
pprint(sys.path)

from autoscript_sdb_microscope_client import SdbMicroscopeClient

