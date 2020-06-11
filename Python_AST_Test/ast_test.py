import torch
import numpy as np
from sklearn import __all__
import sklearn

print("HelloWorld")
sklearn_version = sklearn.__version__
np_version = np.__version__
concat_version = sklearn_version + np_version
print(concat_version)