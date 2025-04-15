import matplotlib.pyplot as plt

def main():
    from datetime import datetime
    from pathlib import Path
    import os

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import tensorflow as tf
    import matplotlib.pyplot as plt

    from pitch_geo import augmentation
    from pitch_geo.dataset import keypoints_dataset, tf_dataloaders
    from pitch_geo.models import models
    from pitch_geo.models.callbacks import LogConfusionMatrixCallback, LogPredictedImages
    from pitch_geo.models.loss import create_teacher_forced_loss
    from pitch_geo.models.metrics import VisiblePrecision, VisibleRecall
    import pitch_geo.vis_utils as vis_utils

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    IMG_SIZE = 240
    BATCH_SIZE = 32
    EPOCHS = 2
    TEACHER_FORCING_WEIGHT = 0.95
    INITIAL_LEARNING_RATE = 1e-3

    train_dataset = keypoints_dataset.get_data(dataset='train')
    train_df, val_df = train_dataset.split(test_size=0.15)

    train_data_builder = tf_dataloaders.KeypointsDatasetBuilder(
        data_frame=train_df, 
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augmentation=augmentation.Sequential([
            augmentation.RandomRotation(angle=10, scale=(0.5, 0.8), p=.5),
            augmentation.RandomTranslation(limit=0.3, p=.5)
        ])
    )
    train_ds = train_data_builder.build()

    val_data_builder = tf_dataloaders.KeypointsDatasetBuilder(
        data_frame=val_df, 
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_ds = val_data_builder.build()

    sample_batch = next(iter(train_ds))
    _ = vis_utils.show_image_with_annotations(
        img=sample_batch[0].numpy()[0, :] / 255.,
        keypoints=sample_batch[1].numpy()[0, :],
        labels=train_dataset.label_map,
        dot_radius=3,
        vis=True
    )

    print(f"Total batches in training set: {len(train_ds)}")
    print(f"Total batches in validation set: {len(val_ds)}")

    model = models.get_model(img_size=IMG_SIZE, num_keypoints=train_dataset.num_keypoints, dropout=0.1)
    model.summary()

    model_base_path = Path('checkpoints')
    checkpoint_template_name = 'cp.ckpt'
    training_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = model_base_path / training_timestamp / checkpoint_template_name
    log_dir = Path('logs') / training_timestamp

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,
        patience=5, 
        min_delta=0.0001,
        min_lr=0.000001,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        min_delta=0.00001,
        verbose=1,
    )

    log_cm = LogConfusionMatrixCallback(model, log_dir, val_ds)
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_cm)

    log_predicted_images = LogPredictedImages(model, log_dir, val_ds, labels=train_dataset.label_map)
    log_images_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_predicted_images, grid_size=4)

    model.compile(
        loss=create_teacher_forced_loss(weight=TEACHER_FORCING_WEIGHT),
        optimizer=tf.keras.optimizers.Adam(INITIAL_LEARNING_RATE),
        metrics=[
            VisiblePrecision(),
            VisibleRecall(),
        ],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            model_checkpoint_callback,
            tensorboard_callback,
            reduce_lr,
            early_stopping,
            cm_callback,
            log_images_callback
        ],
    )

    model = models.get_model(img_size=IMG_SIZE, num_keypoints=train_dataset.num_keypoints, dropout=0.1)
    model.load_weights(model_path)
    model.save(model_path.parent / 'saved_model')

if __name__ == "__main__":
    main()
