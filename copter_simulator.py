import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



used_indexes = []
total_dataset = np.zeros(shape=(200, 29))
random_seed_for_normal_distrib = np.zeros(total_dataset.shape[0] / 4)
random_seed_for_categorical_distrib = np.zeros(total_dataset.shape[0] / 4)
random_seed_for_student_distrib = np.zeros(total_dataset.shape[0] / 4)
random_seed_for_hi_quadrat_distrib = np.zeros(total_dataset.shape[0] / 4)

def full_random_seed(random_seed, class_target):

    for random_index in range(len(total_dataset.shape[0])) / 2:

        random_seed_index = np.random.randint(0, total_dataset.shape[0])
        if (random_seed_index not in random_seed) and (random_seed_index not in used_indexes):

            random_seed[random_index] = random_seed_index
        
        else:
            
            while not (random_seed_index not in random_seed) and (random_seed_index not in used_indexes):
                
                random_seed_index = np.random.randint(0, total_dataset.shape[0])

            random_seed[random_index] = random_seed_index
    
    random_seed.append(class_target)
    random_seed = np.asarray(random_seed)

    return random_seed

class_targets = {
    "normal distrib": 1,
    "student distrib": 2,
    "chisquare distrib": 3,
    "gumbel distrib": 4
}

random_seed_full_1 = full_random_seed(random_seed_for_normal_distrib, class_target=class_targets["normal distrib"])
random_seed_full_2 = full_random_seed(random_seed_for_categorical_distrib, class_target=class_targets["student distrib"])
random_seed_full_3 = full_random_seed(random_seed_for_student_distrib, class_target=class_targets["chisquare distrib"])
random_seed_full_4 = full_random_seed(random_seed_for_hi_quadrat_distrib, class_target=class_targets["gumbel distrib"])

total_dataset[random_seed_full_1] = np.random.normal(0, 1.234, size=total_dataset.shape[1])
total_dataset[random_seed_full_1] = np.random.standard_t(10, size=total_dataset.shape[1])
total_dataset[random_seed_full_1] = np.random.chisquare(45, size=total_dataset.shape[1])
total_dataset[random_seed_full_1] = np.random.gumbel(2.34, 23.234, size=total_dataset.shape[1])

        

def generate_model():

    input_tensor = tf.keras.Inpu(shape=(200, 30))

    first_layer = tf.keras.layers.Dense(200, input_shape=(30, 200), activation="relu")(input_tensor)
    second_layer = tf.keras.layers.Dense(100, activation="relu")(first_layer)
    last_layer = tf.keras.layers.Dense(4, activation="sigmoid")(second_layer)

    model = tf.keras.Model(input_tensor, last_layer)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"]
    )

    return model

def k_fall_learning_algorithm(dataset, block_count=5):

    results_acc = []
    results_error = []
    block_size = dataset.shape[0] // block_count

    for block in block_count:

        val_data = dataset[block * block_size: (block + 1) * block_size, :-1]
        val_targets = dataset[block * block_size: (block + 1) * block_size, -1]

        partial_train_data = np.concatenate(
            [
                dataset[: block * block_size, :-1],
                dataset[block * block_size: (block + 1) * block_size, :-1]
            ], axis=0)
        
        partial_train_targets = np.concatenate(
            [
                dataset[: block * block_size, -1],
                dataset[block * block_size: (block + 1) * block_size, -1]
            ], axis=0)

        model = generate_model()
        model.fit(partial_train_data, partial_train_targets,
                  epochs=20, batch_size=1, verbose=0)
        
        val_accuracy, val_error = model.evaluate(val_data, val_targets, batch_size=1)

        results_acc.append(val_accuracy)
        results_error.append(val_error)

        return results_acc, results_error



acc, error = k_fall_learning_algorithm(dataset=total_dataset)

fig, axis = plt.subplots(nrow=2)
for sample in range(len(axis)):
    axis[sample].plot(range(1, len(axis[sample]) + 1), axis[sample], color="blue", label="acc")
    axis[sample].plot(range(1, len(error[sample])))





    
