import sys
path = '/home/username/myproject'
if path not in sys.path:
    sys.path.insert(0, path)

from myapp import app as application
