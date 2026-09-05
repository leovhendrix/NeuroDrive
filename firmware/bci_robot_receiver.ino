/*
  bci_robot_receiver.ino

  Companion firmware for robot/controller.py's serial protocol.
  Target: Arduino Uno/Nano/Mega or ESP32 with a basic 2-motor driver
  (e.g. L298N / TB6612). Adjust pin numbers to your motor driver.

  Protocol (line-based, newline-terminated ASCII):
    Host -> Board:  "CMD:<letter>\n"   letter in {F,B,L,R,S}
    Host -> Board:  "PING\n"
    Host -> Board:  "ESTOP\n"

    Board -> Host:  "ACK:<letter>\n"
    Board -> Host:  "PONG\n"
    Board -> Host:  "ERR:<reason>\n"

  Safety behavior on the firmware side:
    - If no valid command is received for COMMAND_TIMEOUT_MS, motors stop
      automatically. This is a second, independent safety layer beneath
      the Python safety controller — never rely on the host alone.
    - ESTOP immediately cuts motor output and is only cleared by a fresh
      valid CMD from the host.
*/

#include <Arduino.h>

// ---- Motor driver pins (adjust for your hardware) ----
const int PIN_LEFT_FWD  = 5;
const int PIN_LEFT_BWD  = 6;
const int PIN_RIGHT_FWD = 9;
const int PIN_RIGHT_BWD = 10;

const unsigned long COMMAND_TIMEOUT_MS = 1000;   // must match/exceed host cadence
const unsigned long BAUD_RATE = 115200;

unsigned long lastCommandMillis = 0;
bool emergencyStopped = false;

String inputLine;

void setup() {
  pinMode(PIN_LEFT_FWD, OUTPUT);
  pinMode(PIN_LEFT_BWD, OUTPUT);
  pinMode(PIN_RIGHT_FWD, OUTPUT);
  pinMode(PIN_RIGHT_BWD, OUTPUT);
  stopMotors();

  Serial.begin(BAUD_RATE);
  inputLine.reserve(32);
  lastCommandMillis = millis();
}

void loop() {
  readSerial();
  enforceCommandTimeout();
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      handleLine(inputLine);
      inputLine = "";
    } else if (c != '\r') {
      inputLine += c;
      if (inputLine.length() > 64) inputLine = "";  // guard against garbage
    }
  }
}

void handleLine(const String &line) {
  if (line == "PING") {
    Serial.println("PONG");
    return;
  }
  if (line == "ESTOP") {
    emergencyStopped = true;
    stopMotors();
    Serial.println("ACK:S");
    return;
  }
  if (line.startsWith("CMD:")) {
    char letter = line.length() > 4 ? line.charAt(4) : '\0';
    if (applyCommand(letter)) {
      lastCommandMillis = millis();
      if (letter != 'S') emergencyStopped = false;  // any non-stop cmd clears estop latch
      Serial.print("ACK:");
      Serial.println(letter);
    } else {
      Serial.println("ERR:invalid_command");
    }
    return;
  }
  Serial.println("ERR:unrecognized_message");
}

bool applyCommand(char letter) {
  if (emergencyStopped && letter != 'S') {
    // While e-stopped, only STOP is honored until host sends a fresh valid cmd
    // (host is expected to clear e-stop state before resuming motion).
    stopMotors();
    return true;
  }

  switch (letter) {
    case 'F': driveForward(); return true;
    case 'B': driveBackward(); return true;
    case 'L': turnLeft(); return true;
    case 'R': turnRight(); return true;
    case 'S': stopMotors(); return true;
    default: return false;
  }
}

void enforceCommandTimeout() {
  if (millis() - lastCommandMillis > COMMAND_TIMEOUT_MS) {
    stopMotors();
  }
}

// ---- Motor primitives (adjust speed/pins for your driver board) ----
void stopMotors() {
  analogWrite(PIN_LEFT_FWD, 0);
  analogWrite(PIN_LEFT_BWD, 0);
  analogWrite(PIN_RIGHT_FWD, 0);
  analogWrite(PIN_RIGHT_BWD, 0);
}

void driveForward() {
  analogWrite(PIN_LEFT_FWD, 150);
  analogWrite(PIN_LEFT_BWD, 0);
  analogWrite(PIN_RIGHT_FWD, 150);
  analogWrite(PIN_RIGHT_BWD, 0);
}

void driveBackward() {
  analogWrite(PIN_LEFT_FWD, 0);
  analogWrite(PIN_LEFT_BWD, 150);
  analogWrite(PIN_RIGHT_FWD, 0);
  analogWrite(PIN_RIGHT_BWD, 150);
}

void turnLeft() {
  analogWrite(PIN_LEFT_FWD, 0);
  analogWrite(PIN_LEFT_BWD, 120);
  analogWrite(PIN_RIGHT_FWD, 120);
  analogWrite(PIN_RIGHT_BWD, 0);
}

void turnRight() {
  analogWrite(PIN_LEFT_FWD, 120);
  analogWrite(PIN_LEFT_BWD, 0);
  analogWrite(PIN_RIGHT_FWD, 0);
  analogWrite(PIN_RIGHT_BWD, 120);
}
