import numpy as np
from pathlib import Path
import pandas as pd
from extreme.estimators import real_estimators
from extreme.visualization import real_quantile_plot
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df_real = real_estimators()
    print(df_real)
    # real_quantile_plot()