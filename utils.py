from tensorflow import keras
import mlflow


def run_log_exp(
    run_name: str,
    train_data: tuple,
    val_data: tuple,
    model: keras.Model,
    hyper_params: dict,
):
    """
    The run_log_exp function is used to train a model and log the results of each epoch in corresponding MLFlow run.
    It takes in a run_name, training data, validation data, model and hyperparameters as arguments.
    The function returns the total loss for all epochs.

    :param run_name: str: Name for the run
    :param train_data: tuple: Training data must be in the form ([inputs], [outputs]) which matches model dimensions
    :param val_data: tuple: Validation data must be in the form ([inputs], [outputs]) which matches model dimensions
    :param model: keras.Model: Keras model to use for experiment
    :param hyper_params: dict: Hyperparameters that were used to build/train the model
    :return: The total validation loss for the model
    :doc-author: mcnewcp
    """
    with mlflow.start_run(run_name=run_name):
        history = model.fit(
            train_data[0],
            train_data[1],
            epochs=25,
            validation_data=val_data,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0,
        )
        mlflow.log_params(hyper_params)
        # customization needed here depending on output shape
        total_loss, __, __ = model.evaluate(val_data[0], val_data[1], verbose=0)
        return total_loss
