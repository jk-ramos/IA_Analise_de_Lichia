import io
import os
from datetime import datetime

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.patches import Rectangle
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 20

# Pasta de logs e frames
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", run_name)
frames_dir = os.path.join("frames", run_name)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

validation_generator = val_test_datagen.flow_from_directory(
    "dataset/val",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

class_indices = validation_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}


def plot_to_tensorboard_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=140)
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def create_prediction_figure(images, labels, predictions, index_to_class, epoch, max_images=4):
    max_images = min(max_images, len(images))
    fig, axes = plt.subplots(1, max_images, figsize=(5 * max_images, 6))

    if max_images == 1:
        axes = [axes]

    for i in range(max_images):
        ax = axes[i]
        img = images[i]
        real_label = int(labels[i])

        prob = float(predictions[i][0])
        pred_label = 1 if prob >= 0.5 else 0

        real_class = index_to_class.get(real_label, str(real_label))
        pred_class = index_to_class.get(pred_label, str(pred_label))

        acertou = real_label == pred_label
        border_color = "limegreen" if acertou else "red"
        status_text = "ACERTOU" if acertou else "ERROU"

        confidence = prob if pred_label == 1 else (1.0 - prob)

        ax.imshow(img)
        ax.axis("off")

        rect = Rectangle(
            (0, 0),
            img.shape[1] - 1,
            img.shape[0] - 1,
            linewidth=6,
            edgecolor=border_color,
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.set_title(
            f"{status_text}\n"
            f"Real: {real_class}\n"
            f"Prev: {pred_class}\n"
            f"Confiança: {confidence:.2%}",
            fontsize=11,
            color=border_color,
            fontweight="bold",
            pad=12
        )

    fig.suptitle(
        f"IA analisando lichias - Época {epoch + 1}",
        fontsize=18,
        fontweight="bold"
    )
    plt.tight_layout()
    return fig


class PredictionVideoLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, log_directory, frames_directory, index_to_class, max_images=4):
        super().__init__()
        self.val_generator = val_generator
        self.index_to_class = index_to_class
        self.max_images = max_images
        self.frames_directory = frames_directory
        self.writer = tf.summary.create_file_writer(
            os.path.join(log_directory, "images")
        )
        self.saved_frames = []

    def on_epoch_end(self, epoch, logs=None):
        images, labels = next(self.val_generator)
        images = images[:self.max_images]
        labels = labels[:self.max_images]

        predictions = self.model.predict(images, verbose=0)

        fig = create_prediction_figure(
            images=images,
            labels=labels,
            predictions=predictions,
            index_to_class=self.index_to_class,
            epoch=epoch,
            max_images=self.max_images
        )

        # Salvar no TensorBoard
        tb_image = plot_to_tensorboard_image(fig)
        with self.writer.as_default():
            tf.summary.image(
                name="Predicoes_validacao",
                data=tb_image,
                step=epoch
            )

        # Salvar frame para GIF
        frame_path = os.path.join(self.frames_directory, f"epoch_{epoch + 1:03d}.png")
        imageio.imwrite(frame_path, tb_image[0].numpy())
        self.saved_frames.append(frame_path)

    def on_train_end(self, logs=None):
        gif_path = os.path.join(self.frames_directory, "treinamento_ia_lichia.gif")
        images = [imageio.imread(frame) for frame in self.saved_frames]
        if images:
           mp4_path = os.path.join(self.frames_directory, "treinamento_ia_lichia.mp4")
with imageio.get_writer(mp4_path, fps=1) as writer:
    for frame in images:
        writer.append_data(frame)
print(f"\nVídeo MP4 salvo em: {mp4_path}")

prediction_video_logger_callback = PredictionVideoLoggerCallback(
    val_generator=validation_generator,
    log_directory=log_dir,
    frames_directory=frames_dir,
    index_to_class=index_to_class,
    max_images=4
)

# Model
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=0.0001
    ),
    tensorboard_callback,
    prediction_video_logger_callback
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

model.save("modelo_lichia.keras")

print(f"\nLogs do TensorBoard salvos em: {log_dir}")
print(f"Frames salvos em: {frames_dir}")
print(f"Classes detectadas: {class_indices}")