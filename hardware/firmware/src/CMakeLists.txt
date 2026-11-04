idf_component_register(SRCS "main.c" "freq.c" "ringing.c" "model.c" "nn.cc" "servo.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp_adc driver nvs_flash)
