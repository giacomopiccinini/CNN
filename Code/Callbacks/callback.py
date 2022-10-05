import os
from tensorflow.keras.callbacks  import ModelCheckpoint, EarlyStopping, TensorBoard
from Code.Callbacks.Comet import Comet

def list_callbacks(experiment, train, validation):

    # Retrieve experiment name
    name = experiment.get_name()

    # Create directories for storing results
    os.makedirs(f"Output/{name}/Logs", exist_ok=True)
    os.makedirs(f"Output/{name}/Checkpoints", exist_ok=True)

    # Define checkpoints
    checkpoint = ModelCheckpoint( 
        filepath          = os.path.join(f"Output/{name}/Checkpoints", "{epoch:02d}.hdf5"),
        monitor           = "val_loss",
        verbose           = 1,
        save_best_only    = True,
        save_weights_only = True,
        mode              = "auto",
        save_freq         = "epoch"
    )

    # Define tensorboard
    tensorboard_callback = TensorBoard(log_dir = "Logs")

    # Define early stopping
    earlystopping = EarlyStopping(patience = 10)

    # Define Comet callback
    #comet = Comet(train=train, validation=validation, experiment=experiment)
    comet = Comet(experiment=experiment)

    # Unite callbacks
    callbacks = [checkpoint, tensorboard_callback, earlystopping, comet]

    return callbacks