#include "servo.h"

#include "driver/ledc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define SERVO_MODE LEDC_LOW_SPEED_MODE
#define SERVO_TIMER LEDC_TIMER_0
#define SERVO_CHANNEL LEDC_CHANNEL_0
#define SERVO_RES LEDC_TIMER_14_BIT
#define SERVO_PERIOD_US 20000.0f
#define SERVO_MIN_US 500.0f    // -90 degrees
#define SERVO_MAX_US 2500.0f   // +90 degrees

static float s_angle = 0.0f;

static void write_angle(float a)
{
    if (a < -90.0f) a = -90.0f;
    if (a > 90.0f) a = 90.0f;
    float us = SERVO_MIN_US + (a + 90.0f) / 180.0f * (SERVO_MAX_US - SERVO_MIN_US);
    uint32_t duty = (uint32_t)(us / SERVO_PERIOD_US * ((1 << 14) - 1));
    ledc_set_duty(SERVO_MODE, SERVO_CHANNEL, duty);
    ledc_update_duty(SERVO_MODE, SERVO_CHANNEL);
}

void servo_init(int gpio)
{
    ledc_timer_config_t t = {
        .speed_mode = SERVO_MODE,
        .duty_resolution = SERVO_RES,
        .timer_num = SERVO_TIMER,
        .freq_hz = 50,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&t);
    ledc_channel_config_t c = {
        .gpio_num = gpio,
        .speed_mode = SERVO_MODE,
        .channel = SERVO_CHANNEL,
        .timer_sel = SERVO_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ledc_channel_config(&c);
    s_angle = 0.0f;
    write_angle(s_angle);
    vTaskDelay(pdMS_TO_TICKS(600));
    ledc_stop(SERVO_MODE, SERVO_CHANNEL, 0);
}

void servo_move_to(float target)
{
    const float step = 1.0f;   // degrees per 20 ms frame
    while (s_angle != target) {
        float d = target - s_angle;
        s_angle += (d > step) ? step : (d < -step) ? -step : d;
        write_angle(s_angle);
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    vTaskDelay(pdMS_TO_TICKS(300));            // let it settle
    ledc_stop(SERVO_MODE, SERVO_CHANNEL, 0);   // release: no holding torque needed for a light figure bar
}

void servo_hold(float target, int ms)
{
    servo_move_to(target);
    write_angle(s_angle);                      // signal back on: the servo resists being turned
    vTaskDelay(pdMS_TO_TICKS(ms));
    ledc_stop(SERVO_MODE, SERVO_CHANNEL, 0);
}
