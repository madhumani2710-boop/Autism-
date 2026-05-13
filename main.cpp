#include <Arduino.h>

#define MIC_PIN         34
#define SAMPLE_RATE_MS  10
#define ADC_SAMPLES     4
#define BAUD_RATE       115200

unsigned long lastSample = 0;

void setup() {
    Serial.begin(BAUD_RATE);

    while(!Serial && millis() < 3000); 
    
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db); 
    
    Serial.println("AUTISM_TOY_MIC_READY");
}

void loop() {
    unsigned long now = millis();
    if (now - lastSample >= SAMPLE_RATE_MS) {
        lastSample = now;
        
        long sum = 0;
        for (int i = 0; i < ADC_SAMPLES; i++) {
            sum += analogRead(MIC_PIN);
            delayMicroseconds(250); // Small delay for ADC stabilization
        }
        
        int average = (int)(sum / ADC_SAMPLES);
      
        Serial.println(average);
    }
}
