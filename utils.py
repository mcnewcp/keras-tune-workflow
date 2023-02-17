from tensorflow import keras
import mlflow
import numpy as np
from sklearn.model_selection import KFold


def fit_eval_log(
    run_name: str,
    train_data: tuple,
    val_data: tuple,
    model: keras.Model,
    hyper_params: dict,
    epochs: int = 25,
):
    """
    The fit_eval_log function is used to train a model and log the results of each epoch in corresponding MLFlow run.
    It takes in a run_name, training data, validation data, model and hyperparameters as arguments.
    The function returns the total loss for all epochs.

    :param run_name: str: Name for the run
    :param train_data: tuple: Training data must be in the form ([inputs], [outputs]) which matches model dimensions
    :param val_data: tuple: Validation data must be in the form ([inputs], [outputs]) which matches model dimensions
    :param model: keras.Model: Keras model to use for experiment
    :param hyper_params: dict: Hyperparameters that were used to build/train the model
    :param epochs: int: Maximum number of epochs if early stopping criteria are not met
    :return: The total validation loss for the model
    :doc-author: mcnewcp
    """
    with mlflow.start_run(run_name=run_name):
        history = model.fit(
            train_data[0],
            train_data[1],
            epochs=epochs,
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


def fit_eval_log_cv(
    run_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: keras.Model,
    hyper_params: dict,
    kf: KFold,
    epochs: int = 25,
):
    """
    The fit_eval_log_cv function is used to cross validate a model and log the results of each fold in corresponding MLFlow run.
    It takes in a run_name, training data, model, hyperparameters, and kfold split as arguments.
    The function returns the mean validation loss across all folds.

    :param run_name: str: Name for run in mlflow
    :param X_train: np.ndarray: Full X training data
    :param y_train: np.ndarray: Full corresponding target
    :param model: keras.Model: Keras model to be used
    :param hyper_params: dict: Dictionary of hyperparameters that were used to build the model
    :param kf: KFold: KFold object for splitting data
    :param epochs: int: Maximum number of epochs if early stopping criteria is not met
    :return: The mean validation loss for the cross-validation
    :doc-author: mcnewcp
    """
    with mlflow.start_run(run_name=run_name):
        cv_train_losses = []
        cv_val_losses = []
        k_fold = 1  # keep track of fold number
        for fit, val in kf.split(X_train, y_train):
            with mlflow.start_run(run_name=f"f{k_fold}-{run_name}", nested=True):
                # define train and validation set per fold
                # customize depending on output shape
                train_data = (
                    X_train[fit],
                    {"main_output": y_train[fit], "aux_output": y_train[fit]},
                )
                val_data = (
                    X_train[val],
                    {"main_output": y_train[val], "aux_output": y_train[val]},
                )
                history = model.fit(
                    train_data[0],
                    train_data[1],
                    epochs=epochs,
                    validation_data=val_data,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            patience=5, restore_best_weights=True
                        )
                    ],
                    verbose=0,
                )
                mlflow.log_params(hyper_params)  # log chosen params in child
                # update per fold loss
                train_loss, __, __ = model.evaluate(
                    train_data[0], train_data[1], verbose=0
                )
                val_loss, __, __ = model.evaluate(val_data[0], val_data[1], verbose=0)
                cv_train_losses.append(train_loss)
                cv_val_losses.append(val_loss)
                k_fold += 1  # update fold number
        mlflow.log_params(hyper_params)  # log chosen params in parent
        # log aggregated metrics in parent
        mlflow.log_metrics(
            {
                "train_mean_cv_loss": np.mean(cv_train_losses),
                "val_mean_cv_loss": np.mean(cv_val_losses),
            }
        )
        return np.mean(cv_val_losses)  # return aggregated loss for optimization
