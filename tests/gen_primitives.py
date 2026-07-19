import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning_v2.testkit import regenerate_standard_test
regenerate_standard_test()
print("done")
