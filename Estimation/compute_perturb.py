rhoc_0 = 71.14      # 67.58 (-5%); 71.14; 74.70 (5%)
rhoc_1 = 71.14      # 67.58 (-5%); 71.14; 74.70 (5%)
rhoc_10 = 76.27     #
rhoc_20 = 83.77
rhoc_25 = 86.57  # update    86.57; old 86
rhoc_30 = 91.23
rhoc_40 = 97.53
rhoc_50 = 107.63
rhoc_60 = 116.85
rhoc_70 = 134.65
rhoc_75 = 145.63  # update   145.63; old 142
rhoc_80 = 151.56
rhoc_90 = 183.24
rhoc_99 = 214.06
rhoc_100 = 214.06


val = rhoc_0, rhoc_1, rhoc_10, rhoc_20, rhoc_25, rhoc_30, rhoc_40, rhoc_50, rhoc_60, \
        rhoc_70, rhoc_75, rhoc_80, rhoc_90, rhoc_99, rhoc_100
label = [0, 1, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99, 100]

for i, l in enumerate(label):
    print('rhoc_{0} = {1:.2f}       # {2:.2f} (-5%); {3:.2f}; {4:.2f} (+5%)'.format(l, val[i]*1.05,
                                                                    val[i]*0.95, val[i], val[i]*1.05))