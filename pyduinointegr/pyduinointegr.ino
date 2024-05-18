const int led = LED_BUILTIN;
char msg;


void setup() {
  Serial.begin(9600);
  pinMode(led, OUTPUT);
  digitalWrite(led, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    msg = Serial.read();
    Serial.println(msg);
    
    if (msg == '1'){
      digitalWrite(LED_BUILTIN, HIGH);
      delay(500);
    }

    else if (msg == '0') {
      digitalWrite(led, LOW);
      for(int i = 0; i < 5; i++){
        digitalWrite(led, HIGH);
        delay(100);
        digitalWrite(led, LOW);
        delay(100);
        }
    }

    else {
      digitalWrite(led, LOW);
      delay(10000);
    }
  } else {
    digitalWrite(led, LOW);
  }

}