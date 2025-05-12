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
u = unpacker.Unpacker("OutputFile_cal20250510_Co60", bad_crystals=bad_crystals)
u.make(True)

x_data = u.x_data
y_data = u.y_data
norm = u.norm

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
