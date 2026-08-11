"""
Raspberry Pi Autonomous Driving & Delivery Vehicle
Reconstructed project code based on the original project description.

IMPORTANT
- This is NOT the original 2022-2023 source code.
- GPIO pins, CNN architecture, retraining interval, sensor polarity, and some thresholds
  are reconstructed assumptions and must be checked against the actual hardware.
- Test with wheels lifted from the ground first.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import RPi.GPIO as GPIO


# ============================================================
# 1. Configuration
# ============================================================

IMG_WIDTH = 128
IMG_HEIGHT = 128

BASE_SPEED = 50

BATCH_SIZE = 16
INITIAL_EPOCHS = 10
RETRAIN_EPOCHS = 2

MODEL_PATH = Path("obstacle_model.keras")
DATASET_PATH = Path("dataset")

# Manual-labeled images added before periodic retraining
RETRAIN_IMAGE_COUNT = 100

# Save one labeled frame every N frames while manual labeling is active
SAVE_FRAME_INTERVAL = 10

# CNN classes reconstructed from remembered training structure
CLASS_NAMES = [
    "normal",
    "obstacle_1",
    "obstacle_2",
    "obstacle_3",
    "obstacle_4",
    "obstacle_5",
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# 2. GPIO Pin Map
#    Example BCM pin numbers - verify before use.
# ============================================================

# Ultrasonic sensor
TRIG = 23
ECHO = 24

# Two IR line-tracking sensors
IR_LEFT = 17
IR_RIGHT = 22

# Motor driver
LEFT_IN1 = 5
LEFT_IN2 = 6
LEFT_PWM_PIN = 12

RIGHT_IN1 = 16
RIGHT_IN2 = 20
RIGHT_PWM_PIN = 13

PWM_FREQUENCY = 1000


# ============================================================
# 3. IR Sensor Polarity
#    Reconstructed assumption.
#
# The actual module may output the reverse logic.
# Change these values if the vehicle reacts backward.
# ============================================================

WHITE = 1
BLACK = 0


# ============================================================
# 4. Dataset Preparation
# ============================================================

def prepare_dataset_folders():
    DATASET_PATH.mkdir(exist_ok=True)

    for class_name in CLASS_NAMES:
        (DATASET_PATH / class_name).mkdir(exist_ok=True)


def count_dataset_images():
    total = 0

    for class_name in CLASS_NAMES:
        folder = DATASET_PATH / class_name

        total += sum(
            1 for f in folder.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    return total


# ============================================================
# 5. CNN Model
# ============================================================

def create_model():
    """
    Image-classification CNN.

    Reconstructed class concept:
        normal
        obstacle_1
        obstacle_2
        obstacle_3
        obstacle_4
        obstacle_5
    """

    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Rescaling(1.0 / 255.0),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_or_create_model():
    if MODEL_PATH.exists():
        print(f"[CNN] Loading model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        trained = True
    else:
        print("[CNN] No saved model found. Creating a new model.")
        model = create_model()
        trained = False

    return model, trained


def load_training_dataset():
    """
    Folder names are used as class names.
    The class order is forced to match CLASS_NAMES.
    """

    return tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        class_names=CLASS_NAMES,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )


def train_model(model, epochs):
    total_images = count_dataset_images()

    print()
    print("========================================")
    print("CNN TRAINING")
    print(f"Dataset images : {total_images}")
    print(f"Epochs         : {epochs}")
    print("========================================")

    # A real project should use more data.
    # This threshold only prevents accidental training on nearly empty folders.
    if total_images < 20:
        print("[CNN] Not enough labeled images. Training skipped.")
        return False

    dataset = load_training_dataset()

    model.fit(
        dataset,
        epochs=epochs,
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"[CNN] Model saved: {MODEL_PATH}")

    return True


def preprocess_frame(frame):
    image = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    return image


def classify_frame(model, frame):
    image = preprocess_frame(frame)
    prediction = model.predict(image, verbose=0)[0]

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return class_index, confidence


# ============================================================
# 6. Camera/CNN Speed Rule
#
# Remembered speed concept:
# normal     -> 50
# obstacle 1 -> 40
# obstacle 2 -> 30
# obstacle 3 -> 20
# obstacle 4 -> 10
# obstacle 5 -> STOP
# ============================================================

def speed_by_camera(obstacle_class):
    if obstacle_class >= 5:
        return 0
    if obstacle_class == 4:
        return 10
    if obstacle_class == 3:
        return 20
    if obstacle_class == 2:
        return 30
    if obstacle_class == 1:
        return 40

    return 50


# ============================================================
# 7. Ultrasonic Speed Rule
#
# Remembered concept:
# > 50 cm -> 50
# 40-50   -> 40
# 30-40   -> 30
# 20-30   -> 20
# 10-20   -> 10
# <=10    -> STOP
# ============================================================

def speed_by_distance(distance_cm):
    if distance_cm <= 10:
        return 0
    if distance_cm <= 20:
        return 10
    if distance_cm <= 30:
        return 20
    if distance_cm <= 40:
        return 30
    if distance_cm <= 50:
        return 40

    return 50


def determine_safe_speed(camera_speed, ultrasonic_speed):
    """
    Reconstructed sensor-fusion rule:
    use the more conservative speed limit.
    """

    return min(camera_speed, ultrasonic_speed)


# ============================================================
# 8. GPIO / Sensor Functions
# ============================================================

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

    GPIO.setup(IR_LEFT, GPIO.IN)
    GPIO.setup(IR_RIGHT, GPIO.IN)

    for pin in (
        LEFT_IN1,
        LEFT_IN2,
        LEFT_PWM_PIN,
        RIGHT_IN1,
        RIGHT_IN2,
        RIGHT_PWM_PIN,
    ):
        GPIO.setup(pin, GPIO.OUT)


def get_distance():
    """
    Measure obstacle distance with an HC-SR04-style ultrasonic sensor.

    Returns:
        distance in centimeters
        999.0 on timeout
    """

    GPIO.output(TRIG, False)
    time.sleep(0.000002)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.monotonic()
    pulse_end = pulse_start

    timeout_at = time.monotonic() + 0.03

    while GPIO.input(ECHO) == 0:
        pulse_start = time.monotonic()

        if time.monotonic() > timeout_at:
            return 999.0

    timeout_at = time.monotonic() + 0.03

    while GPIO.input(ECHO) == 1:
        pulse_end = time.monotonic()

        if time.monotonic() > timeout_at:
            return 999.0

    elapsed = pulse_end - pulse_start

    # Speed of sound: about 34300 cm/s.
    # Divide by 2 because the pulse travels to the object and back.
    distance = (elapsed * 34300.0) / 2.0

    return round(distance, 2)


def read_line_sensors():
    left_ir = GPIO.input(IR_LEFT)
    right_ir = GPIO.input(IR_RIGHT)

    return left_ir, right_ir


# ============================================================
# 9. Two-Sensor Line Tracking
#
# Road concept:
# - black road surface
# - white lane boundary on the left and right
#
# IR variable resistors on the sensor boards were physically tuned
# to improve white/black discrimination under real driving conditions.
# ============================================================

def line_tracking_control(left_ir, right_ir, base_speed):
    """
    Convert two IR sensor states into left/right motor speed.

    base_speed:
        maximum safe speed determined by camera/CNN + ultrasonic sensor

    NOTE:
        This steering rule is reconstructed.
        If actual sensor placement was reversed, LEFT/RIGHT correction
        may also need to be reversed.
    """

    if base_speed <= 0:
        return 0, 0, "STOP"

    # Both sensors see black road -> vehicle remains inside lane boundaries
    if left_ir == BLACK and right_ir == BLACK:
        return base_speed, base_speed, "FORWARD"

    # Left sensor reaches white lane -> correct to the right
    if left_ir == WHITE and right_ir == BLACK:
        left_speed = base_speed
        right_speed = int(base_speed * 0.4)

        return left_speed, right_speed, "RIGHT_CORRECTION"

    # Right sensor reaches white lane -> correct to the left
    if left_ir == BLACK and right_ir == WHITE:
        left_speed = int(base_speed * 0.4)
        right_speed = base_speed

        return left_speed, right_speed, "LEFT_CORRECTION"

    # Both sensors detect white.
    # Reconstructed fail-safe: stop instead of guessing.
    return 0, 0, "LINE_LOST"


# ============================================================
# 10. Motor Control
# ============================================================

def clamp_speed(speed):
    return max(0, min(100, int(speed)))


def motor_forward(left_pwm, right_pwm, left_speed, right_speed):
    left_speed = clamp_speed(left_speed)
    right_speed = clamp_speed(right_speed)

    GPIO.output(LEFT_IN1, GPIO.HIGH)
    GPIO.output(LEFT_IN2, GPIO.LOW)

    GPIO.output(RIGHT_IN1, GPIO.HIGH)
    GPIO.output(RIGHT_IN2, GPIO.LOW)

    left_pwm.ChangeDutyCycle(left_speed)
    right_pwm.ChangeDutyCycle(right_speed)


def stop_car(left_pwm, right_pwm):
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)

    GPIO.output(LEFT_IN1, GPIO.LOW)
    GPIO.output(LEFT_IN2, GPIO.LOW)

    GPIO.output(RIGHT_IN1, GPIO.LOW)
    GPIO.output(RIGHT_IN2, GPIO.LOW)


# ============================================================
# 11. Manual Labeling / Continuous Data Collection
# ============================================================

def save_labeled_frame(frame, label_index):
    class_name = CLASS_NAMES[label_index]
    folder = DATASET_PATH / class_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = folder / f"{timestamp}.jpg"

    cv2.imwrite(str(filename), frame)

    return filename


# ============================================================
# 12. UI Overlay
# ============================================================

def draw_status(
    frame,
    class_name,
    confidence,
    distance,
    left_ir,
    right_ir,
    camera_speed,
    ultrasonic_speed,
    safe_speed,
    left_motor_speed,
    right_motor_speed,
    direction,
    manual_label,
    new_image_count,
):
    label_text = "OFF" if manual_label is None else CLASS_NAMES[manual_label]

    lines = [
        f"CNN: {class_name} ({confidence:.2f})",
        f"Distance: {distance:.1f} cm",
        f"IR L/R: {left_ir}/{right_ir}",
        f"Camera speed: {camera_speed}",
        f"Ultrasonic speed: {ultrasonic_speed}",
        f"Safe speed: {safe_speed}",
        f"Motor L/R: {left_motor_speed}/{right_motor_speed}",
        f"Direction: {direction}",
        f"Manual label: {label_text}",
        f"New data: {new_image_count}",
    ]

    y = 28

    for line in lines:
        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        y += 25


# ============================================================
# 13. Main
# ============================================================

def main():
    prepare_dataset_folders()
    setup_gpio()

    left_pwm = GPIO.PWM(LEFT_PWM_PIN, PWM_FREQUENCY)
    right_pwm = GPIO.PWM(RIGHT_PWM_PIN, PWM_FREQUENCY)

    left_pwm.start(0)
    right_pwm.start(0)

    model, model_trained = load_or_create_model()

    # If labeled images already exist but the model does not,
    # perform an initial training pass.
    if not model_trained and count_dataset_images() >= 20:
        model_trained = train_model(model, INITIAL_EPOCHS)

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        stop_car(left_pwm, right_pwm)
        GPIO.cleanup()
        raise RuntimeError("Camera could not be opened.")

    manual_label = None
    frame_count = 0
    new_image_count = 0

    print()
    print("======================================================")
    print(" Raspberry Pi Autonomous Vehicle - Reconstructed")
    print("======================================================")
    print("0 : label NORMAL")
    print("1 : label OBSTACLE_1")
    print("2 : label OBSTACLE_2")
    print("3 : label OBSTACLE_3")
    print("4 : label OBSTACLE_4")
    print("5 : label OBSTACLE_5")
    print("x : stop manual labeling")
    print("t : retrain CNN now")
    print("q : quit")
    print("======================================================")
    print()

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("[CAMERA] Frame capture failed.")
                stop_car(left_pwm, right_pwm)
                continue

            frame_count += 1

            # ------------------------------------------------
            # Camera / CNN
            # ------------------------------------------------
            if model_trained:
                obstacle_class, confidence = classify_frame(model, frame)
                camera_speed = speed_by_camera(obstacle_class)
                class_name = CLASS_NAMES[obstacle_class]
            else:
                # Until a model has actually been trained, do not trust random CNN output.
                obstacle_class = 0
                confidence = 0.0
                camera_speed = BASE_SPEED
                class_name = "UNTRAINED"

            # ------------------------------------------------
            # Ultrasonic distance
            # ------------------------------------------------
            distance = get_distance()
            ultrasonic_speed = speed_by_distance(distance)

            # ------------------------------------------------
            # Camera + ultrasonic speed fusion
            # ------------------------------------------------
            safe_speed = determine_safe_speed(
                camera_speed,
                ultrasonic_speed,
            )

            # ------------------------------------------------
            # Two IR sensors decide left/right correction
            # ------------------------------------------------
            left_ir, right_ir = read_line_sensors()

            (
                left_motor_speed,
                right_motor_speed,
                direction,
            ) = line_tracking_control(
                left_ir,
                right_ir,
                safe_speed,
            )

            # ------------------------------------------------
            # Motor output
            # ------------------------------------------------
            if left_motor_speed == 0 and right_motor_speed == 0:
                stop_car(left_pwm, right_pwm)
            else:
                motor_forward(
                    left_pwm,
                    right_pwm,
                    left_motor_speed,
                    right_motor_speed,
                )

            # ------------------------------------------------
            # Manual labeled image collection
            # ------------------------------------------------
            if (
                manual_label is not None
                and frame_count % SAVE_FRAME_INTERVAL == 0
            ):
                filename = save_labeled_frame(frame, manual_label)
                new_image_count += 1
                print(f"[DATA] Saved: {filename}")

            # ------------------------------------------------
            # Periodic retraining
            # ------------------------------------------------
            if new_image_count >= RETRAIN_IMAGE_COUNT:
                print("[CNN] New labeled-data threshold reached.")
                print("[SAFETY] Vehicle stopped during retraining.")

                stop_car(left_pwm, right_pwm)

                if train_model(model, RETRAIN_EPOCHS):
                    model_trained = True

                new_image_count = 0

            # ------------------------------------------------
            # Display
            # ------------------------------------------------
            draw_status(
                frame=frame,
                class_name=class_name,
                confidence=confidence,
                distance=distance,
                left_ir=left_ir,
                right_ir=right_ir,
                camera_speed=camera_speed,
                ultrasonic_speed=ultrasonic_speed,
                safe_speed=safe_speed,
                left_motor_speed=left_motor_speed,
                right_motor_speed=right_motor_speed,
                direction=direction,
                manual_label=manual_label,
                new_image_count=new_image_count,
            )

            cv2.imshow(
                "Autonomous Driving - Reconstructed",
                frame,
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key in (
                ord("0"),
                ord("1"),
                ord("2"),
                ord("3"),
                ord("4"),
                ord("5"),
            ):
                manual_label = int(chr(key))
                print(
                    f"[LABEL] {CLASS_NAMES[manual_label]}"
                )

            elif key == ord("x"):
                manual_label = None
                print("[LABEL] OFF")

            elif key == ord("t"):
                print("[CNN] Manual retraining requested.")
                stop_car(left_pwm, right_pwm)

                if train_model(model, RETRAIN_EPOCHS):
                    model_trained = True

                new_image_count = 0

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt.")

    finally:
        print("[SAFETY] Vehicle stopping.")

        stop_car(left_pwm, right_pwm)

        left_pwm.stop()
        right_pwm.stop()

        camera.release()
        cv2.destroyAllWindows()

        GPIO.cleanup()

        print("[INFO] Camera / PWM / GPIO cleanup complete.")


if __name__ == "__main__":
    main()
