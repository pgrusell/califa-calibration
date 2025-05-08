import unpacker
import calibrate_NN
import test_NN

# Get the data from the root file
u = unpacker.Unpacker('60Co')
u.make(False)
x_data = u.x_data

# Load and the model
cal = calibrate_NN.Calibration()
cal.load_model()

# Show the results
test = test_NN.Test(cal, x_data)
test.plot_results()

