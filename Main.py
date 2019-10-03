
import numpy as np
from sklearn import preprocessing
import tensorflow as tf


def main():

    ####Task: Create a machine Learning Algorithm that can predict if a customer will buy again
    raw_csv_data = np.loadtxt('Audiobooks-data.csv', delimiter=',')

    # The inputs are all columns in the csv, except for the first one [:,0]
    # (which is just the arbitrary customer IDs that bear no useful information),
    # and the last one [:,-1] (which is our targets)

    unscaled_inputs_all = raw_csv_data[:, 1:-1]
    targets_all = raw_csv_data[:, -1]

    #### Shuffle the data

    shuffled_indices = np.arange(unscaled_inputs_all.shape[0])
    np.random.shuffle(shuffled_indices)
    # Use the shuffled indices to shuffle the inputs and targets.
    unscaled_inputs_all = unscaled_inputs_all[shuffled_indices]
    targets_all = targets_all[shuffled_indices]

    ####Balance the dataset

    num_one_targets = int(np.sum(targets_all))
    zero_targets_counter = 0
    # We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
    indices_to_remove = []

    for i in range(targets_all.shape[0]):
        if targets_all[i] == 0:
            zero_targets_counter += 1
            if zero_targets_counter > num_one_targets:
                indices_to_remove.append(i)

    unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
    targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)
    scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
    #### Shuffle the data

    shuffled_indices = np.arange(scaled_inputs.shape[0])
    np.random.shuffle(shuffled_indices)
    shuffled_inputs = scaled_inputs[shuffled_indices]
    shuffled_targets = targets_equal_priors[shuffled_indices]


    #### Split the dataset into TRAIN,VALIDATION and TEST

    samples_count = shuffled_inputs.shape[0]

    # Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
    train_samples_count = int(0.8 * samples_count)
    validation_samples_count = int(0.1 * samples_count)

    test_samples_count = samples_count - train_samples_count - validation_samples_count

    train_inputs = shuffled_inputs[:train_samples_count]
    train_targets = shuffled_targets[:train_samples_count]

    validation_inputs = shuffled_inputs[train_samples_count:train_samples_count + validation_samples_count]
    validation_targets = shuffled_targets[train_samples_count:train_samples_count + validation_samples_count]

    test_inputs = shuffled_inputs[train_samples_count + validation_samples_count:]
    test_targets = shuffled_targets[train_samples_count + validation_samples_count:]

    print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
    print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
    print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

    #### save the data sets in npz

    np.savez('Audiobooks_data_train.npz', inputs=train_inputs, targets=train_targets)
    np.savez('Audiobooks_data_validation.npz', inputs=validation_inputs, targets=validation_targets)
    np.savez('Audiobooks_data_test.npz', inputs=test_inputs, targets=test_targets)


    #### load the Data

    npz=np.load('Audiobooks_data_train.npz')
    train_inputs=npz['inputs'].astype(np.float)
    train_targets=npz['targets'].astype(np.int)

    npz=np.load('Audiobooks_data_validation.npz')
    validation_inputs,validation_targets=npz['inputs'].astype(np.float),npz['targets'].astype(np.int)

    npz = np.load('Audiobooks_data_test.npz')
    test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

    #### Model

    input_size = 10
    output_size = 2
    hidden_layer_size = 50

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    batch_size=100
    max_epochs=100
    early_stopping=tf.keras.callbacks.EarlyStopping(patience=2)

    model.fit(train_inputs,  # train inputs
              train_targets,  # train targets
              batch_size=batch_size,  # batch size
              epochs=max_epochs,  # epochs that we will train for (assuming early stopping doesn't kick in)
              # callbacks are functions called by a task when a task is completed
              # task here is to check if val_loss is increasing
              callbacks=[early_stopping],  # early stopping
              validation_data=(validation_inputs, validation_targets),  # validation data
              verbose=2  # making sure we get enough information about the training process
              )


    #### Test the model

    test_loss,test_accuracy=model.evaluate(test_inputs,test_targets)
    print(f'test loss {test_loss}  test accuracy {test_accuracy}')


if __name__ == '__main__':
    main()