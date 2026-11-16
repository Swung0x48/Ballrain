import sys
import numpy as np
from collections import Counter
sys.path.insert(0, '.')
import pf
frames, ev = pf.parse(sys.argv[1] + "/frames.txt", want={"T", "K"})
T = [(fr["idx"], fr["T"]["cur"], fr["T"]["next"], fr["T"]["last"], fr["T"]["cc"]) for fr in frames if "T" in fr]
last = np.array([t[3] for t in T]); cur = np.array([t[1] for t in T])
d = np.diff(last)
print("env last_psi step counter:", Counter(np.round(d, 6)).most_common(6))
print("cur - last:", Counter(np.round(cur - last, 6)).most_common(4), "cc:", Counter(round(t[4], 7) for t in T).most_common(3))
