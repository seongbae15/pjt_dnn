from gc import callbacks
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def train_model(
    model,
    train_x,
    train_y,
    batch_size,
    learning_rate=0.001,
    epochs=100,
    callback_fn=None,
):
    optimizer = Adam(learning_rate)
    loss_fn = "mean_squared_error"
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    history = model.fit(
        train_x[0].numpy(),
        train_y[0].numpy(),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(train_x[1].numpy(), train_y[1].numpy()),
        verbose=1,
        callbacks=callback_fn,
    )
    return history


def set_train_callback():
    CP = ModelCheckpoint(
        filepath="Models/{epoch:03d}-{loss:.4f}-{accuracy:.4f}.hdf5",
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    LR = ReduceLROnPlateau(
        monitor="loss", factor=0.8, patience=3, verbose=1, min_lr=1e-8
    )
    CALLBACK = [CP, LR]
    return CALLBACK
