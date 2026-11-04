// The weather house's on-board neural network, run with TensorFlow Lite Micro.
//
// A small int8 network (three 1D convolutions over the last 24 hours, then two dense layers; 3777
// weights, 12 KB) reads each spectrum band's hourly placing against its own week for the last 24 hours
// (model_nn_input in model.h) and gives the predicted cleanliness: 0 for the dirtiest half hour of the
// week, 1 for the cleanest. It was trained on the PC (hardware/tools/tinyml_train.py) and is fixed in
// the firmware: no clock, no internet, no learning on the board.
#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Loads the model and runs the built-in self-test; false if either fails.
bool nn_init(void);

// in: 24 x 13 values, row h = h hours ago, column = band. Returns the predicted cleanliness, or NAN.
float nn_predict(const float *in);

// Figure position -1 (power station fully out) to +1 (turbine fully out) from a prediction.
float nn_position(float pred);

#ifdef __cplusplus
}
#endif
