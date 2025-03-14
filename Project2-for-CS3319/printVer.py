import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import metric_learn

def printVer():
    print('numpy version:', np.__version__)
    print('pandas version:', pd.__version__)
    print('sklearn version:', sklearn.__version__)
    print('metric_learn version:', metric_learn.__version__)

if __name__ == '__main__':
    printVer()