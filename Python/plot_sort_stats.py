import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_file(path):
    print("Loading " + path)
    return pd.read_csv(path, header=None, delimiter=";").values


constant32 = load_file("constant32.csv")
constant64 = load_file("constant64.csv")

plt.plot(constant32[:, 1], constant32[:, 2], label="32 bits")
plt.plot(constant64[:, 1], constant64[:, 2], label="64 bits")
plt.xlabel("Num Elements")
plt.ylabel("B/s")
plt.xscale("log")
plt.legend()
plt.show()

variable32 = load_file("variable32.csv")
variable64 = load_file("variable64.csv")

plt.plot(variable32[:, 0], variable32[:, 2], label="32 bits")
plt.plot(variable64[:, 0], variable64[:, 2], label="64 bits")
plt.xlabel("Num bits")
plt.ylabel("B/s")
plt.legend()
plt.show()
