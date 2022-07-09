from tensorflow.keras.optimizers import SGD

def train_model(model, train_x, train_y, batch_size, learning_rate):
    optimizer = SGD(learning_rate, momentum=0.9)
    loss_fn = "mean_squared_error"
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    epochs = 100
    history = model.fit(train_x[0], train_y[0], batch_size=batch_size, epochs=epochs,
                        validation_data=(train_x[1], train_y[1]), verbose=1)
    