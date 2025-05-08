import unpacker
import calibrate_NN
import test_NN

import matplotlib.pyplot as plt

bad_crystals = [
    1744, 1989, 2339, 2360, 2485, 4502, 4533, 4551, 4676, 4687,
    4825, 4883, 4904, 4964, 4977, 4978, 4979, 4983, 4990, 4991,
    5009, 5012, 5014, 5015, 5018, 5021, 5024, 5027, 5030, 5057,
    5071
]

# Unpack the data from the data directory
u = unpacker.Unpacker("calData", bad_crystals=bad_crystals)
u.make(True)

x_data = u.x_data
y_data = u.y_data
norm = u.norm

# Create the NN
epochs = 61
cal = calibrate_NN.Calibration(x_data, y_data, norm, epochs)
cal.make()

# Test the NN
test = test_NN.Test(cal, cal.x_test, cal.y_test)
test.make_test()

# Try on bad crystals
u = unpacker.Unpacker("calData")
u.make(True)

for i, (key, val) in enumerate(u.db.items()):

    if key == 805:

        x = u.x_data[i]
        y = u.y_data[i]

        fig, ax = plt.subplots()

        ax.plot(u.common_bins, x)

        yp = cal.predict_value(x)

        ax.vlines(y[0], 0, max(x), linestyle='--', color='green')
        ax.vlines(y[1], 0, max(x), linestyle='--', color='red')
