import os

D1 = [f"../data/day1/{x}" for x in os.listdir("../data/day1")]
D2 = [f"../data/day2/{x}" for x in os.listdir("../data/day2")]


def make_pairings(i):
    i = [x for x in i if "67_near" in x or "nt1_middle" in x]
    D = {}
    for x in i:
        sp = x.split("/")[-1].split(".")[0].split("_")[-1]
        if not (sp in D):
            D[sp] = []
        D[sp].append(x)
    o = []
    for x in D.values():
        ii = 0 if "nt1_middle" in x[0] else 1
        ti = 0 if ii == 1 else 1
        o.append((x[ii], x[ti]))
    return o


FILES = make_pairings(D1) + make_pairings(D2)

if __name__ == "__main__":
    print(FILES)
