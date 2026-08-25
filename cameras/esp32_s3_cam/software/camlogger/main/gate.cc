/* TFLite Micro animal-gate inference. Ops registered = exactly what the
 * converter emits for MobileNetV1+Dense+Softmax (see wavesharecam
 * LESSONS.md #31: register what the model contains, verified by dumping
 * ops from the .tflite, or the interpreter dies at load). */
#include "gate.h"
#include "gate_model_data.h"

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "gate";

namespace {
constexpr int kArenaSize = 600 * 1024;
const tflite::Model *model;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input_t;
uint8_t *arena;
}

extern "C" bool gate_init(void)
{
    model = tflite::GetModel(g_gate_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "bad model schema %lu",
                 (unsigned long)model->version());
        return false;
    }
    arena = (uint8_t *)heap_caps_malloc(kArenaSize,
                                        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) {
        ESP_LOGE(TAG, "arena alloc failed");
        return false;
    }
    /* exactly the ops the converter emitted (dumped from the .tflite) */
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddAdd();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddMean();
    resolver.AddSoftmax();
    static tflite::MicroInterpreter interp(model, resolver, arena,
                                           kArenaSize);
    interpreter = &interp;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return false;
    }
    input_t = interpreter->input(0);
    ESP_LOGI(TAG, "gate ready (%d B model, %d B arena used)",
             g_gate_model_data_len,
             (int)interpreter->arena_used_bytes());
    return true;
}

extern "C" float gate_score(const int8_t *img)
{
    if (!interpreter) return -1.0f;
    memcpy(input_t->data.int8, img, GATE_IMG * GATE_IMG * 3);
    if (interpreter->Invoke() != kTfLiteOk) return -1.0f;
    TfLiteTensor *out = interpreter->output(0);
    /* class 1 = animal; dequantize int8 output */
    return (out->data.int8[1] - out->params.zero_point) * out->params.scale;
}
