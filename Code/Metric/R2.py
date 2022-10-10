import keras.backend as K


def r2(y_true, y_pred):

    """Coefficient of Determination implemented in Kears"""

    # Compute sum of squares of residuals
    SS_res = K.sum(K.square(y_true - y_pred))

    # Compute total sum of square
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    # Add epsilon to prevent divergences
    return 1 - SS_res / (SS_tot + K.epsilon())
