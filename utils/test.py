import unpacker
import calibrate_NN
import test_NN

# Unpack the data from the data directory
u = unpacker.Unpacker("calData")
u.make(True)

x_data = u.x_data
y_data = u.y_data
norm = u.norm

# Create the NN
epochs = 70
cal = calibrate_NN.Calibration(x_data, y_data, norm, epochs)
cal.make()

# Test the NN
test = test_NN.Test(cal, cal.x_test, cal.y_test)
test.make_test()
