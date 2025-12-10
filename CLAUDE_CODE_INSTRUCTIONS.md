# GridSeis ESP32-S3 Firmware Development Instructions

## Project Overview

Build an ESP32-S3 firmware that replicates the GridSeis Python pipeline for offline carbon intensity prediction from grid frequency measurements. The device should:

1. Passively detect 50Hz mains frequency from ambient electromagnetic fields (no mains connection)
2. Sample at 1Hz, accumulate 30-minute windows (1800 samples)
3. Perform FFT analysis to extract frequency-domain features
4. Run XGBoost inference to predict carbon intensity
5. Store/display predictions locally

## Reference Implementation

The Python implementation is in the GridSeis repository:
- `helpers.py`: `perform_fft_analysis()` - FFT feature extraction
- `2_modelling.py`: XGBoost training, feature engineering with rolling windows

### Current Python Pipeline

```
1Hz samples → 30-min window (1800 pts) → FFT → 810 frequency bins → rolling features (1h/3h/6h min/max) → XGBoost → carbon intensity prediction
```

### Key Parameters
- Sampling rate: 1 Hz
- Window size: 1800 samples (30 minutes)
- FFT output: ~810 bins (frequencies > 0.05 Hz)
- Rolling windows: 1h, 3h, 6h with min/max aggregations
- Current feature count: ~5600 (too many for embedded)

## Architecture Decision: Feature Reduction Required

The full Python model uses ~5600 features which is impractical for ESP32. Two approaches:

### Option A: Reduced Feature Model (Recommended)
1. Analyse feature importance from trained XGBoost model
2. Select top 50-100 most important features
3. Retrain model with reduced feature set
4. Export to C using m2cgen

### Option B: Frequency Band Aggregation
1. Instead of 810 individual FFT bins, aggregate into ~20-30 frequency bands
2. Compute band power (sum of squared magnitudes) for each band
3. Simpler rolling features (e.g., just 1h mean)
4. Retrain model on aggregated features

## Development Phases

### Phase 1: Frequency Capture Module

**Goal**: Accurately measure 50Hz grid frequency using ambient EM field pickup.

**Reference**: bertrik/GridFrequency project uses IQ correlation method.

**Implementation**:
```
Hardware:
- ESP32-S3 DevKit (e.g., ESP32-S3-WROOM-1)
- Floating ADC pin (no connection) or short wire antenna (10-30cm)
- Optional: SD card for data logging

Algorithm:
1. Sample ADC at ~1000 Hz for 1 second
2. Correlate samples with reference 50Hz sine/cosine (IQ demodulation)
3. Calculate phase from I and Q values
4. Track phase difference between consecutive seconds
5. Derive instantaneous frequency (target: 1mHz resolution)
```

**Files to create**:
- `src/freq_capture.h` / `src/freq_capture.cpp`
- Use ESP-IDF ADC continuous mode for reliable sampling
- Output: one frequency reading per second (float, in Hz)

**Test criteria**:
- Frequency readings should be in range 49.8-50.2 Hz
- Readings should be stable (< 5mHz std dev over quiet periods)
- Compare against National Grid frequency data for validation

### Phase 2: FFT Feature Extraction

**Goal**: Replicate Python `perform_fft_analysis()` on ESP32.

**Implementation**:
```c
// Accumulate 1800 frequency samples (30 minutes)
float freq_buffer[1800];
int sample_count = 0;

// When buffer full:
// 1. Remove mean (centre the signal)
// 2. Apply Hanning window
// 3. Perform FFT using ESP-DSP library
// 4. Extract magnitude spectrum
// 5. Filter bins > 0.05 Hz (or aggregate into bands)
```

**ESP-DSP FFT**:
```c
#include "esp_dsp.h"

// For 1800-point FFT, pad to 2048 (next power of 2)
// Or use 1024-point FFT on downsampled/windowed data

float fft_input[2048];  // Real values, zero-padded
float fft_output[2048]; // Complex output (interleaved)

dsps_fft2r_fc32(fft_input, 2048);
dsps_bit_rev_fc32(fft_input, 2048);
```

**Files to create**:
- `src/fft_features.h` / `src/fft_features.cpp`
- `src/feature_config.h` - defines which bins/bands to extract

**Feature reduction strategy** (implement one):

```c
// Option 1: Select specific FFT bins based on Python feature importance
const int IMPORTANT_BINS[] = {12, 45, 67, 89, ...}; // From analysis
const int NUM_FEATURES = 50;

// Option 2: Aggregate into frequency bands
typedef struct {
    float low_freq;
    float high_freq;
} FreqBand;

const FreqBand BANDS[] = {
    {0.05, 0.08},
    {0.08, 0.12},
    // ... 20-30 bands total
};
```

### Phase 3: Rolling Window Features

**Goal**: Maintain history for temporal features without excessive memory.

**Implementation**:
```c
// Circular buffer for feature history
// 6 hours = 12 windows of 30 minutes each
#define HISTORY_WINDOWS 12
#define NUM_BASE_FEATURES 50  // After reduction

float feature_history[HISTORY_WINDOWS][NUM_BASE_FEATURES];
int history_index = 0;

// After each FFT extraction:
// 1. Store new features in circular buffer
// 2. Compute rolling min/max over available history
// 3. Assemble final feature vector for inference
```

**Memory budget**:
- 12 windows × 50 features × 4 bytes = 2.4 KB (easily fits in SRAM)

**Files to create**:
- `src/rolling_features.h` / `src/rolling_features.cpp`

### Phase 4: XGBoost Model Export and Inference

**Goal**: Run trained model on ESP32.

**Python side** (add to GridSeis repo):
```python
# scripts/export_model.py

import m2cgen as m2c
import pickle

# Load trained model
with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Export to C code
code = m2c.export_to_c(model)

with open("firmware/src/model.c", "w") as f:
    f.write(code)

# Also export feature names/indices for reference
with open("firmware/src/feature_names.h", "w") as f:
    f.write("// Feature order for model input\n")
    for i, name in enumerate(feature_names):
        f.write(f"// {i}: {name}\n")
```

**ESP32 side**:
```c
// model.h
double score(double *features);

// main inference call
float features[NUM_FEATURES];
// ... populate features from FFT + rolling ...
double carbon_intensity = score(features);
```

**Files to create**:
- `src/model.c` - auto-generated from m2cgen
- `src/model.h` - declares score() function
- `src/inference.cpp` - orchestrates feature assembly and prediction

### Phase 5: Main Application Loop

```c
void app_main() {
    // Initialisation
    init_adc_continuous();
    init_fft();
    init_sd_card();  // Optional
    
    float freq_buffer[1800];
    int sample_idx = 0;
    
    while (true) {
        // Every second: capture frequency
        float freq = capture_frequency();
        freq_buffer[sample_idx++] = freq;
        
        // Every 30 minutes: run inference
        if (sample_idx >= 1800) {
            float base_features[NUM_BASE_FEATURES];
            extract_fft_features(freq_buffer, base_features);
            
            update_rolling_history(base_features);
            
            float full_features[NUM_TOTAL_FEATURES];
            assemble_features(full_features);
            
            double carbon = score(full_features);
            
            log_prediction(carbon);
            display_prediction(carbon);  // If display attached
            
            sample_idx = 0;
        }
        
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
```

## Project Structure

```
gridseis/
├── firmware/
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults
│   ├── main/
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   ├── components/
│   │   ├── freq_capture/
│   │   │   ├── CMakeLists.txt
│   │   │   ├── freq_capture.h
│   │   │   └── freq_capture.c
│   │   ├── fft_features/
│   │   │   ├── CMakeLists.txt
│   │   │   ├── fft_features.h
│   │   │   └── fft_features.c
│   │   ├── rolling_features/
│   │   │   ├── CMakeLists.txt
│   │   │   ├── rolling_features.h
│   │   │   └── rolling_features.c
│   │   └── model/
│   │       ├── CMakeLists.txt
│   │       ├── model.h
│   │       └── model.c          # Auto-generated
│   └── README.md
├── scripts/
│   ├── export_model.py          # Export XGBoost to C
│   ├── analyse_features.py      # Feature importance analysis
│   └── validate_esp_output.py   # Compare ESP vs Python outputs
└── docs/
    └── hardware_setup.md
```

## Dependencies

**ESP-IDF components**:
- `esp_adc` - ADC continuous mode
- `esp_dsp` - FFT functions (add to idf_component.yml)

**idf_component.yml**:
```yaml
dependencies:
  espressif/esp-dsp: "^1.3.0"
```

## Validation Strategy

1. **Frequency capture validation**:
   - Log raw frequency readings to SD card
   - Compare against National Grid published frequency data
   - Target correlation > 0.95

2. **FFT validation**:
   - Feed identical 1800-sample windows to Python and ESP32
   - Compare extracted features (should match within floating point tolerance)

3. **End-to-end validation**:
   - Run ESP32 predictions alongside Python predictions on same time period
   - Compare carbon intensity outputs
   - Acceptable if R² > 0.8 vs Python model

## Hardware BOM (Minimal)

| Component | Example Part | Purpose |
|-----------|--------------|---------|
| ESP32-S3 DevKit | ESP32-S3-WROOM-1 | Main MCU |
| Wire antenna | 20cm insulated wire | 50Hz pickup |
| SD card module | Generic SPI | Data logging (optional) |
| OLED display | SSD1306 128x64 | Show predictions (optional) |

## Notes for Claude Code

- Use ESP-IDF v5.x (not Arduino framework) for better ADC control
- The ESP32-S3 ADC2 channels are preferred (ADC1 conflicts with WiFi)
- For IQ demodulation, precompute sin/cos lookup tables
- m2cgen output can be large; if > 500KB, need to reduce model complexity
- Test frequency capture first before building full pipeline
- Consider OTA update capability for model updates

## Getting Started Commands

```bash
# Set up ESP-IDF
. $HOME/esp/esp-idf/export.sh

# Create project
idf.py create-project gridseis-firmware
cd gridseis-firmware

# Add ESP-DSP dependency
idf.py add-dependency "espressif/esp-dsp^1.3.0"

# Configure for ESP32-S3
idf.py set-target esp32s3

# Build and flash
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

## Success Criteria

1. Device continuously measures grid frequency with < 5mHz noise
2. FFT features match Python implementation within 1% tolerance
3. Carbon intensity predictions correlate with Python model (R² > 0.8)
4. Runs indefinitely on mains power without memory leaks
5. 30-minute prediction cycle completes in < 5 seconds
