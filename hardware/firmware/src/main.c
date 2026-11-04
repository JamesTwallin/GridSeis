// GridSeis weather house: an ESP32-S3 listens to the mains hum on a wire, measures the
// grid frequency once a second, and turns a servo so that a wind turbine comes out of
// the door when the grid is cleaner than usual for the week and a power station comes
// out when it is dirtier.
//
// The figures follow an on-board neural network (nn.h, TensorFlow Lite Micro, int8) that reads
// 13 bands of the frequency's spectrum, each placed against its own week, hour by hour over the
// last 24 hours (model.h). Everything runs on the board; it never connects to anything. Until the
// network has two days of its own history, the figures follow the weighted sum of the same bands
// (from about 36 hours), and before that the older one-band rule: loud ringing at 4 to 10 second
// periods means a light, windy, low-carbon grid.
//
// Wiring (see hardware/README.md):
//   antenna: a 10-30 cm wire on GPIO1 (ADC1 channel 0), far end left free
//   servo signal -> GPIO4, servo 5V -> 5V pin, servo GND -> GND
//
// Serial output at 115200 baud: one line every second with frequency, signal amplitude
// and the chip clock, and two lines every 5 minutes: the ringing with its weekly rank, and the
// model's prediction with the angle the servo was sent to.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_adc/adc_continuous.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs.h"
#include "nvs_flash.h"

#include "freq.h"
#include "model.h"
#include "nn.h"
#include "ringing.h"
#include "servo.h"

#define ANTENNA_CHANNEL ADC_CHANNEL_0   // GPIO1 on the ESP32-S3; ADC1 keeps working with WiFi on
#define SERVO_GPIO 4
#define ANGLE_MAX 60.0f                 // degrees either side; matches the enclosure geometry check
#define CLEAN_SIGN (-1.0f)              // -1: clean turns the bar clockwise (turbine out of the right door)
#define SEAT_HOLD_S 30                  // seconds the servo holds dead centre at start-up, to seat the bar
#define FRAME_BYTES 256
#define HIST_SAVE_EVERY 12              // save the week of history to flash every hour
#define NN_MIN_HOURS 6                  // hours of the board's own readings before the network sets the figures;
                                        // in the test its match was 0.18 at 3 to 6 h, 0.38 at 6 to 12 h, 0.74 after a week

static const char *TAG = "gridseis";
static adc_continuous_handle_t s_adc;
static volatile bool s_overflow = false;
static uint16_t s_block[FREQ_FS];
static freq_state_t s_freq;
static ring_state_t s_ring;
static model_state_t s_model;
static bool s_nn_ok = false;

static bool IRAM_ATTR on_overflow(adc_continuous_handle_t h, const adc_continuous_evt_data_t *e, void *u)
{
    s_overflow = true;
    return false;
}

static void adc_start(void)
{
    // The ADC is never stopped: stopping and restarting it left a gap in the samples and
    // a +0.3 Hz spike in one reading every five minutes, which swamped the ringing. The
    // pool holds 8 s of samples, enough to cover the spectrum and the servo move.
    adc_continuous_handle_cfg_t hcfg = {.max_store_buf_size = 65536, .conv_frame_size = FRAME_BYTES};
    ESP_ERROR_CHECK(adc_continuous_new_handle(&hcfg, &s_adc));
    adc_digi_pattern_config_t pattern = {
        .atten = ADC_ATTEN_DB_12,
        .channel = ANTENNA_CHANNEL,
        .unit = ADC_UNIT_1,
        .bit_width = SOC_ADC_DIGI_MAX_BITWIDTH,
    };
    adc_continuous_config_t cfg = {
        .pattern_num = 1,
        .adc_pattern = &pattern,
        .sample_freq_hz = FREQ_FS,
        .conv_mode = ADC_CONV_SINGLE_UNIT_1,
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
    };
    ESP_ERROR_CHECK(adc_continuous_config(s_adc, &cfg));
    adc_continuous_evt_cbs_t cbs = {.on_pool_ovf = on_overflow};
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(s_adc, &cbs, NULL));
    ESP_ERROR_CHECK(adc_continuous_start(s_adc));
}

typedef struct {
    int32_t head, n;
    float hist[RING_HIST];
} saved_hist_t;

static void hist_load(void)
{
    nvs_handle_t h;
    if (nvs_open("gridseis", NVS_READONLY, &h) != ESP_OK) return;
    static saved_hist_t s;
    size_t len = sizeof(s);
    if (nvs_get_blob(h, "hist", &s, &len) == ESP_OK && len == sizeof(s) && s.n >= 0 && s.n <= RING_HIST) {
        memcpy(s_ring.hist, s.hist, sizeof(s.hist));
        s_ring.hist_head = s.head % RING_HIST;
        s_ring.hist_n = s.n;
        ESP_LOGI(TAG, "loaded %d ringing values (%.1f days) from flash", (int)s.n, s.n * RING_HOP / 86400.0);
    }
    nvs_close(h);
}

// The model's state lives in its own flash partition ("model", see partitions.csv).
static void model_load(void)
{
    nvs_handle_t h;
    if (nvs_open_from_partition("model", "gridseis", NVS_READONLY, &h) != ESP_OK) return;
    size_t len = sizeof(s_model);
    if (nvs_get_blob(h, "state", &s_model, &len) != ESP_OK || len != sizeof(s_model) ||
        s_model.version != MODEL_STATE_VERSION || s_model.kept_n < 0 || s_model.kept_n > MODEL_THIN_N ||
        s_model.rank_n < 0 || s_model.rank_n > MODEL_RANKS_N) {
        model_init(&s_model);
    } else {
        ESP_LOGI(TAG, "loaded the spectrum model's history (%.1f days) from flash", model_history_days(&s_model));
    }
    nvs_close(h);
}

static void model_save(void)
{
    nvs_handle_t h;
    if (nvs_open_from_partition("model", "gridseis", NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_blob(h, "state", &s_model, sizeof(s_model));
    nvs_commit(h);
    nvs_close(h);
}

static void hist_save(void)
{
    nvs_handle_t h;
    if (nvs_open("gridseis", NVS_READWRITE, &h) != ESP_OK) return;
    static saved_hist_t s;
    s.head = s_ring.hist_head;
    s.n = s_ring.hist_n;
    memcpy(s.hist, s_ring.hist, sizeof(s.hist));
    nvs_set_blob(h, "hist", &s, sizeof(s));
    nvs_commit(h);
    nvs_close(h);
}

// Seconds whose samples were taken while the servo signal was on are marked invalid, in
// case the servo's pulses reach the antenna. The ADC keeps running throughout, so the
// phase chain is never broken. Returns the last second number to treat as invalid.
static uint32_t every_five_minutes(uint32_t seconds)
{
    static int saves = 0;
    if (s_ring.filled < RING_WIN) {
        ESP_LOGI(TAG, "ringing: collecting the first 30 minutes (%d s so far)", s_ring.filled);
        return 0;
    }
    int64_t t0 = esp_timer_get_time();
    float ringing, bands[MODEL_NBANDS];
    bool ok = ring_compute(&s_ring, &ringing);
    if (ok) ok = ring_compute_bands(&s_ring, MODEL_NBANDS, MODEL_K_LO, MODEL_K_HI, MODEL_HUM_K_LO, MODEL_HUM_K_HI, bands);
    float position;
    float pred = model_update(&s_model, bands, ok, &position);   // counts the slot even when it is invalid
    static float nn_in[24 * MODEL_NBANDS];
    bool nn_ready = ok && s_nn_ok && s_model.count >= NN_MIN_HOURS * 12 && model_nn_input(&s_model, nn_in);
    float nn_pred = nn_ready ? nn_predict(nn_in) : NAN;
    float nn_pos = nn_position(nn_pred);
    int compute_ms = (int)((esp_timer_get_time() - t0) / 1000);
    if (!ok) {
        ESP_LOGW(TAG, "ringing: not enough valid seconds in the last 30 minutes (antenna near a mains cable?)");
        printf("MODEL,nan,nan,nan,nan,%.2f,none,%d\n", model_history_days(&s_model), compute_ms);
        if (++saves % HIST_SAVE_EVERY == 0) { hist_save(); model_save(); }
        return 0;
    }
    float rank = ring_percentile(&s_ring, ringing);   // rank against the week before adding this value
    ring_hist_push(&s_ring, ringing);
    float angle = 0.0f;
    const char *source = "none";
    if (!isnan(nn_pos)) {
        angle = CLEAN_SIGN * nn_pos * ANGLE_MAX;
        source = "network";
    } else if (!isnan(position)) {
        angle = CLEAN_SIGN * position * ANGLE_MAX;
        source = "weighted_sum";
    } else if (s_ring.hist_n >= RING_MIN_HIST) {
        angle = CLEAN_SIGN * (rank - 0.5f) * 2.0f * ANGLE_MAX;
        source = "ringing";
    }
    printf("RING,%.5f,%.3f,%.1f,%d\n", ringing, rank, angle, s_ring.hist_n);
    printf("MODEL,%.4f,%.3f,%.4f,%.3f,%.2f,%s,%d\n", nn_pred, nn_pos, pred, position, model_history_days(&s_model),
           source, compute_ms);
    t0 = esp_timer_get_time();
    servo_move_to(angle);
    int64_t moving_us = esp_timer_get_time() - t0;
    if (++saves % HIST_SAVE_EVERY == 0) { hist_save(); model_save(); }
    // Samples taken during the move land in the next block and the following ones; skip
    // one more for margin.
    return seconds + 2 + (uint32_t)((moving_us + 999999) / 1000000);
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    err = nvs_flash_init_partition("model");
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase_partition("model");
        nvs_flash_init_partition("model");
    }
    freq_init(&s_freq);
    ring_init(&s_ring);
    model_init(&s_model);
    hist_load();
    model_load();
    servo_init(SERVO_GPIO);
    // Start-up: hold dead centre so the bar can be pushed on parallel to the front wall,
    // then a wave out to both limits shows the servo, the direction and the clearances.
    printf("SEAT: servo held at centre for %d s - push the bar on now, arms parallel to the front wall\n",
           SEAT_HOLD_S);
    servo_hold(0.0f, SEAT_HOLD_S * 1000);
    printf("SEAT: done, sweeping to both limits\n");
    servo_move_to(-ANGLE_MAX);
    servo_move_to(ANGLE_MAX);
    servo_move_to(0.0f);
    // The network starts after the seat hold, so its self-test prints while a serial monitor is
    // attached (the USB port only reappears a second or two after a reset).
    s_nn_ok = nn_init();
    printf("NN: %s\n", s_nn_ok ? "on-board network ready" : "on-board network FAILED its self-test; using the weighted sum");
    adc_start();
    printf("FREQ,seconds,frequency_hz,amplitude_counts,valid,chip_time_us\n");
    printf("RING,ringing_mhz,weekly_rank,angle_deg,history_values\n");
    printf("MODEL,network_cleanliness,network_position,sum_cleanliness,sum_position,model_history_days,figures_follow,"
           "compute_ms\n");

    static uint8_t frame[FRAME_BYTES];
    int fill = 0;
    uint32_t seconds = 0;
    uint32_t skip_until = 0;   // seconds up to this number were sampled while the servo moved
    while (true) {
        uint32_t got = 0;
        esp_err_t r = adc_continuous_read(s_adc, frame, FRAME_BYTES, &got, 2000);
        if (r != ESP_OK) continue;
        if (s_overflow) {        // samples were lost: the current second is broken
            s_overflow = false;
            fill = 0;
            freq_reset_phase(&s_freq);
        }
        for (uint32_t i = 0; i + SOC_ADC_DIGI_RESULT_BYTES <= got; i += SOC_ADC_DIGI_RESULT_BYTES) {
            adc_digi_output_data_t *p = (adc_digi_output_data_t *)&frame[i];
            if (p->type2.unit != ADC_UNIT_1 || p->type2.channel != ANTENNA_CHANNEL) continue;
            s_block[fill++] = (uint16_t)p->type2.data;
            if (fill < FREQ_FS) continue;
            fill = 0;
            bool ok = freq_process_second(&s_freq, s_block, FREQ_FS);
            seconds++;
            if (seconds <= skip_until) ok = false;
            ring_push_second(&s_ring, s_freq.freq_hz, ok);
            printf("FREQ,%lu,%.5f,%.1f,%d,%lld\n", (unsigned long)seconds, s_freq.freq_hz, s_freq.amplitude, ok,
                   (long long)esp_timer_get_time());
            if (seconds % RING_HOP == 0) skip_until = every_five_minutes(seconds);
        }
    }
}
