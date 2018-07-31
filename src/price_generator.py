import numpy as np
import pandas as pd

def random_walk(a, h, l, start, end, freq):
    index = pd.date_range(start=start, end=end, freq=freq)
    num_steps = len(index)
    r = []
    r.append(a)
    c = a
    for i in range(num_steps - 1):
        dt = np.random.uniform(-1, 1)
        if((c + dt < l) or (c + dt > h)):
            dt = -dt
        c += dt
        r.append(round(c, 2))
    return pd.Series(np.array(r), index=index)


def random_price():
    start_day = np.random.choice(pd.date_range(start='2000-01-01', end='2018-08-01', freq='D'))
    end_day = start_day + np.timedelta64(127, 'D')
    start = np.random.randint(10, 200)
    low = max(5, start - np.random.randint(0, 200))
    high = min(200, start + np.random.randint(0, 200))
    return pd.Series(random_walk(start, high, low, start_day, end_day, 'D'))