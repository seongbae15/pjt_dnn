from tensorflow.keras.optimizers import SGD, Adam


def train_model(model, train_x, train_y, batch_size, learning_rate=0.001, epochs=100):
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
    )
    return history

