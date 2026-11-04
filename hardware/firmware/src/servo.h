// SG90-class hobby servo on one GPIO, 50 Hz PWM from the LEDC peripheral.
// The servo is moved slowly and then released, so it does not buzz and its current
// spikes do not land on the antenna input while the ADC is listening.
#pragma once

void servo_init(int gpio);
void servo_move_to(float angle_deg);   // -90 to +90, moves at about 50 degrees a second, then releases
void servo_hold(float angle_deg, int ms);   // go to an angle and hold it firmly for ms, then release
