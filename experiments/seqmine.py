import pickle
from prefixspan import PrefixSpan

with open("../data/objects/paths", "rb") as f:
    paths = pickle.load(f)

ps = PrefixSpan(paths)
freqs = ps.frequent(2)

with open("../data/objects/freqs", "wb") as f:
    pickle.dump(freqs, f)
