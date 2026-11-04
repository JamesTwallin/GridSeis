#include "nn.h"

#include <math.h>
#include <stdio.h>

#include "nn_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr int kArenaSize = 32 * 1024;
alignas(16) uint8_t s_arena[kArenaSize];
tflite::MicroInterpreter *s_interp = nullptr;
}  // namespace

extern "C" float nn_predict(const float *in)
{
    if (!s_interp) return NAN;
    TfLiteTensor *x = s_interp->input(0);
    for (int i = 0; i < NN_HOURS * NN_BANDS; i++) {
        float q = roundf(in[i] / NN_IN_SCALE) + NN_IN_ZERO;
        x->data.int8[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    if (s_interp->Invoke() != kTfLiteOk) return NAN;
    return ((float)s_interp->output(0)->data.int8[0] - NN_OUT_ZERO) * NN_OUT_SCALE;
}

extern "C" float nn_position(float pred)
{
    if (isnan(pred)) return NAN;
    float p = pred < NN_P_MID ? (pred - NN_P_MID) / (NN_P_MID - NN_P_LO) : (pred - NN_P_MID) / (NN_P_HI - NN_P_MID);
    return p < -1.0f ? -1.0f : (p > 1.0f ? 1.0f : p);
}

extern "C" bool nn_init(void)
{
    const tflite::Model *model = tflite::GetModel(g_nn_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("NN: model schema %lu, library %d\n", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddExpandDims();
    resolver.AddFullyConnected();
    resolver.AddPack();
    resolver.AddReshape();
    resolver.AddShape();
    resolver.AddStridedSlice();
    static tflite::MicroInterpreter interp(model, resolver, s_arena, kArenaSize);
    if (interp.AllocateTensors() != kTfLiteOk) {
        printf("NN: could not allocate tensors\n");
        return false;
    }
    s_interp = &interp;
    // Self-test: three inputs from 2024-2026 with the outputs the PC's TensorFlow Lite gave.
    bool ok = true;
    for (int k = 0; k < NN_SELFTEST_N; k++) {
        float got = nn_predict(g_nn_selftest_in[k]);
        bool match = fabsf(got - g_nn_selftest_out[k]) <= 1.5f * NN_OUT_SCALE;
        printf("NN: self-test %d: expected %.4f, got %.4f %s\n", k, g_nn_selftest_out[k], got, match ? "ok" : "MISMATCH");
        ok = ok && match;
    }
    printf("NN: arena used %u of %d bytes\n", (unsigned)interp.arena_used_bytes(), kArenaSize);
    if (!ok) s_interp = nullptr;
    return ok;
}
