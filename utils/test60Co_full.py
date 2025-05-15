import unpacker
import calibrate_NN
import test_NN

import matplotlib.pyplot as plt
import numpy as np

bad_crystals = [
    1744, 1989, 2339, 2360, 2485, 4502, 4533, 4551, 4676, 4687,
    4825, 4883, 4904, 4964, 4977, 4978, 4979, 4983, 4990, 4991,
    5009, 5012, 5014, 5015, 5018, 5021, 5024, 5027, 5030, 5057,
    5071
]

# Unpack the data from the data directory
u = unpacker.Unpacker("OutputFile_cal20250510_Co60",
                      bad_crystals=bad_crystals)
u.make(True)

x_data_unzoom = u.x_data
y_data_unzoom = u.y_data
norm_unzoom = u.norm


# Unpack the data from the data directory
u2 = unpacker.Unpacker("calData", bad_crystals=bad_crystals)
u2.make(True)

x_data_zoom = u2.x_data
y_data_zoom = u2.y_data
norm_zoom = u2.norm

# Unpack the data from the data directory
u3 = unpacker.Unpacker("OutputFile_cal20250510_Co60",
                       bad_crystals=bad_crystals, min_bins=300, max_bins=1400)
u3.make(True)

x_data_zoom3 = u3.x_data
y_data_zoom3 = u3.y_data
norm_zoom3 = u3.norm

# Unpack the data from the data directory
u4 = unpacker.Unpacker("OutputFile_cal20250510_Co60",
                       bad_crystals=bad_crystals, noise=True)
u4.make(True)

x_data_4 = u4.x_data
y_data_4 = u4.y_data
norm_4 = u4.norm


# Unpack the data from the data directory
u5 = unpacker.Unpacker("calData", bad_crystals=bad_crystals, noise=True)
u5.make(True)

x_data_zoom5 = u5.x_data
y_data_zoom5 = u5.y_data
norm_zoom5 = u5.norm

# Unpack the data from the data directory
u6 = unpacker.Unpacker("OutputFile_cal20250510_Co60",
                       bad_crystals=bad_crystals, min_bins=300, max_bins=1400, noise=True)
u6.make(True)

x_data_zoom6 = u6.x_data
y_data_zoom6 = u6.y_data
norm_zoom6 = u6.norm

x_data = np.concat([x_data_unzoom, x_data_zoom, x_data_zoom3,
                   x_data_4, x_data_zoom5, x_data_zoom6])
y_data = np.concat([y_data_unzoom, y_data_zoom, y_data_zoom3,
                   y_data_4, y_data_zoom5, y_data_zoom6])
norm = np.concat([norm_unzoom, norm_zoom, norm_zoom3,
                 norm_4, norm_zoom5, norm_zoom6])


# Create the NN
epochs = 150
cal = calibrate_NN.Calibration(x_data, y_data, norm, epochs)
cal.make()

# Test the NN
test = test_NN.Test(cal, cal.x_test, cal.y_test)
test.make_test()


y_pred = cal.predict_value(x_data)

# Predict (also with bad crystals)

u = unpacker.Unpacker("OutputFile_cal20250510_Co60")
u.make(True)

for i in range(len(x_data)):

    x = x_data[i]
    y = y_pred[i]

    fig, ax = plt.subplots()
    ax.plot(u.common_bins, x)
    ax.vlines(y[0], 0, max(x), linestyles='--', color='gray')
    ax.vlines(y[1], 0, max(x), linestyles='--', color='gray')

    plt.savefig(f'../results/crystal_{i}.pdf')
    plt.close(fig)

print("Fin")
