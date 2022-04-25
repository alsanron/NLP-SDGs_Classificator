sdgs = ["np", "zh", "gh", "qe", "ge", "cw", "ae", "dw", "iii", "ri", "sc", "rcp", "ca", "lbw", "lol", "pjsi", "pftg"]

for (ii, sdg) in zip(range(1,18), sdgs):
    for year in range(2016,2022):
        name = "{:02d}_{}_{}.txt".format(ii, sdg, year)
        f = open(name, 'w')
        f.close()