import math

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def opt_learn_rate_plot(model, X, y, lr0=10**-5, lr1=10, s=500, batch_size=64):
    """Plot learning rate vs loss, to find optimal learning rate through elbow trick, learning rate at elbow / 10.
    
    Model is compiled with ADAM optimizer and MSE loss.
    For further information, see page 325f. in Hands-On Machinea Learning with Scikit-Learn, Keras & TensorFlow.

    Args:
        model (tensorflow.keras.models.Model): Model
        X (numpy.array): Training data
        y (numpy.array): Training data target
        lr0 (float, optional): Start of tested learning rate. Defaults to 10**-5.
        lr1 (float, optional): End of tested learning rate. Defaults to 10.
        s (int, optional): Number of steps. Defaults to 500.
        batch_size (int, optional): Batch Size. Defaults to 64.

    Returns:
        None: Image plotting learning rate vs log(loss)
    """
    tmp_model = tf.keras.models.clone_model(model)
    tmp_model.compile(
        optimizer=Adam(),
        loss='mean_squared_error'
    )
    def learning_rate_scheduler(lr0=10**(-5), lr1=10, s=50):
        def exp_increade_fn(epoch):
            return lr0 * math.exp(math.log(lr1/lr0) / s) ** (epoch)
        return exp_increade_fn

    history = tmp_model.fit(
        X, y,
        epochs=s,
        batch_size=batch_size,
        verbose=1,
        steps_per_epoch=1,
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler(lr0, lr1, s)),
            tf.keras.callbacks.TerminateOnNaN()
        ]
    )
    del tmp_model
    plt.plot(
        history.history['lr'],
        history.history['loss'],
        # pd.DataFrame(history.history).ewm(span=100).mean().loc[:, 'loss']
    )
    plt.semilogy()
    plt.xlabel('lr')
    plt.ylabel('loss')
    plt.show()