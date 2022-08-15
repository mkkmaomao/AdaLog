import pickle
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from official.nlp import optimization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
# import keras.backend as K
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.pos_encoding = positional_encoding(max_len,
                                                embed_dim)

    def call(self, x):
        # x = self.token_emb(x)
        seq_len = tf.shape(x)[1]
        # print(maxlen)
        x += self.pos_encoding[:, :seq_len, :]
        # positions = tf.range(start=0, limit=maxlen, delta=1)
        # positions = self.pos_emb(positions)
        # print(x.shape, positions.shape)
        # x = self.token_emb(x)
        return x




def transformer_classifer(input_size, loss_object, optimizer, dropout=0.1):
    inputs = layers.Input(shape=(max_len, embed_dim))
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    embedding_layer = PositionEmbedding(100, 2000, embed_dim)
    # print(inputs.shape)
    x = embedding_layer(inputs)
    # print(x.shape)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss_object, metrics=['accuracy'],
                  optimizer=optimizer)
    return model



class BatchGenerator(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print(self.batch_size)
        dummy = np.zeros(shape=(embed_dim,))
        x = self.X[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.X))]
        X = np.zeros((len(x), max_len, embed_dim))
        Y = np.zeros((len(x), 2))
        item_count = 0
        for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.X))):
            x = self.X[i]
            if len(x) > max_len:
                x = x[-max_len:]
            x = np.pad(np.array(x), pad_width=((max_len - len(x), 0), (0, 0)), mode='constant',
                       constant_values=0)
            X[item_count] = np.reshape(x, [max_len, embed_dim])
            Y[item_count] = self.Y[i]
            item_count += 1
        # return X[:], Y[:, 0]
        # print("in getitem:")
        # print("x shape: ", np.shape(X))
        # print("y shape: ", np.shape(Y))
        # print("y: ", Y)
        # return X[:], Y[:, 0]
        return X[:], Y[:]


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                      epoch_num, model_name=None):
    # learning_rate = CustomSchedule(768)

    # optim = tf.keras.optimizers.Adam(learning_rate)

    optim = Adam()
    epochs = epoch_num
    steps_per_epoch = num_train_samples
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    def weighted_binary_crossentropy(y_true, y_pred):
        # Initialized a tensor of weights such that negative entries will receive a higher weight (=60), hence missing them will lead to a greater penalty.
        weights = (tf.math.abs(y_true - 1) * 55.) + 1.
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce

    loss_object = tfa.losses.SigmoidFocalCrossEntropy()
    # loss_object = weighted_binary_crossentropy
    # loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # label_smoothing=True

    model = transformer_classifer(768, loss_object, optimizer)

    # model.load_weights(load_file_epoch20)

    print(model.summary())

    # checkpoint
    filepath = model_name
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stop]

    # class_weight = {0: 245., 1: 1.}

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=int(num_train_samples / batch_size),
                        epochs=epoch_num,
                        verbose=1,
                        validation_data=validate_generator,
                        validation_steps=int(num_val_samples / batch_size),
                        workers=16,
                        max_queue_size=32,
                        callbacks=callbacks_list,
                        shuffle=True
                        # class_weight=class_weight
                        )

    # model.fit(x=training_generator.X, y=training_generator.Y, validation_data=validate_generator, batch_size=64, epochs=20, verbose=1,
    #           max_queue_size=32, callbacks=callbacks_list, shuffle=True)
    # model.save('./weights/epoch={}.h5'.format(epoch_num))
    return model

def train_generator_shuffle(training_generator, num_train_samples, batch_size,
                      epoch_num, model_name=None):
    # learning_rate = CustomSchedule(768)

    # optim = tf.keras.optimizers.Adam(learning_rate)

    optim = Adam()
    epochs = epoch_num
    steps_per_epoch = num_train_samples * 0.9
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model = transformer_classifer(768, loss_object, optimizer)

    print(model.summary())

    # checkpoint
    filepath = model_name
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stop]


    model.fit(np.asarray(training_generator.X).astype(np.int32), np.asarray(training_generator.Y).astype(np.int32),
                        steps_per_epoch=int(num_train_samples * 0.9 / batch_size),
                        epochs=epoch_num,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks_list,
                        shuffle=True
                        )

    return model


def train(X, Y, epoch_num, batch_size, model_file=None):

    X, Y = shuffle(X, Y)
    n_samples = len(X)
    # -------- previous version for other datasets --------
    train_x, train_y = X[:int(n_samples * 90 / 100)], Y[:int(n_samples * 90 / 100)]
    val_x, val_y = X[int(n_samples * 90 / 100):], Y[int(n_samples * 90 / 100):]

    print("train_Y :", train_y[:10])
    training_generator, num_train_samples = BatchGenerator(train_x, train_y, batch_size), len(train_x)
    validate_generator, num_val_samples = BatchGenerator(val_x, val_y, batch_size), len(val_x)
    print("Number of training samples: {0} - Number of validating samples: {1}".format(num_train_samples,
                                                                                       num_val_samples))

    model = train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                            epoch_num, model_name=model_file)

    # -------- change -------- shuffle train and val set
    # training_generator, num_train_samples = BatchGenerator(X, Y, batch_size), len(X)
    # model = train_generator_shuffle(training_generator, num_train_samples, batch_size, epoch_num, model_name=model_file)

    # test_model(model, tx, ty, batch_size)
    # return model


def test_model(x, y, batch_size, epochs):
    import random
    SEED = 1
    random.seed(SEED)
    random.shuffle(x)
    random.seed(SEED)
    random.shuffle(y)

    # x, y = shuffle(x, y)
    x, y = x[: len(x) // batch_size * batch_size], y[: len(y) // batch_size * batch_size]
    # x, y = x[: int(len(x))], y[: int(len(y))]
    print("y: ", len(y))
    test_loader = BatchGenerator(x, y, batch_size)
    print("test loader.Y: ",len(y))
    init_lr = 3e-4
    steps_per_epoch = len(x)
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    loss_object = tfa.losses.SigmoidFocalCrossEntropy()
    # loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # label_smoothing=True (remove it when using Spirit_5m dataset)

    model = transformer_classifer(768, loss_object, optimizer)
    model.load_weights(weight_file)

    # model = tf.keras.models.load_model('hdfs_transformer.hdf5')
    prediction = model.predict(test_loader, steps=(len(x) // batch_size), workers=16, max_queue_size=32,
                                         verbose=1)
    # prediction = model.predict(np.asarray(test_loader.X).astype(np.int32), steps=(len(x) // batch_size), workers=16, max_queue_size=32,
    #                                      verbose=1)
    # model.predict_generator

    prediction = np.argmax(prediction, axis=1) # index0->normal index1->anormalous
    print("prediction :", prediction)
    y = y[:len(prediction)]
    y_true = np.argmax(test_loader.Y, axis=1)
    # pre = prediction.tolist()
    # tru = y_true.tolist()
    print("y length: ", len(y_true))
    print("prediction: ", prediction)
    # print("prediction 1st anomaly: ",pre.index(1))
    print("y-true: ", y_true)
    # print("y-true 1st anomaly: ", tru.index(1))
    print("predictions: ",Counter(prediction))
    print("true labels: ", Counter(y_true))

    def calculate_metric():
        TP, TN, FN, FP = 0, 0, 0, 0
        for i in range(len(prediction)):
            if (prediction[i] == y_true[i]):
                if (prediction[i] == 1):
                    TP = TP + 1
                else:
                    TN = TN + 1
            else:
                if ((prediction[i] == 1) & (y_true[i] == 0)):
                    FP = FP + 1
                elif ((prediction[i] == 0) & (y_true[i] == 1)):
                    FN = FN + 1
        print("tp: ", TP)
        print("fn: ", FN)
        print("fp: ", FP)
        print("tn: ", TN)
        Spec = TN / (TN + FP)
        Recall = TP / (TP + FN)
        print("Recall: ", Recall)
        Precision = TP / (TP + FP)
        print("Precision: ", Precision)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        print("F1: ", F1)
        return F1
    F1 = calculate_metric()
    return F1
    # report = classification_report(np.array(y), prediction)
    # report = classification_report(test_loader.Y, prediction)
    # print(report)

def load_prob(x_tr):
    with open("../clusters/{}/{}/per={}/probability/k={}&{}_d={}.pickle".format(dataset, ws, percent, k1, k2, dim), 'rb') as f:
        probs = pickle.load(f)

    prob_array = []
    print("len x_tr", len(x_tr))
    print("len probs", len(probs))
    for i in range(len(x_tr)):
        prob_array.append((probs.get(i), 1 - probs.get(i)))

    return x_tr, prob_array

def load_test(x_te, y_te):

    binary_array = [] # p(normal, abnormalous)
    for i in range(len(y_te)):
        # print("y_te[i]", y_te[i])
        if (y_te[i] == 0):
            binary_array.append((1, 0))
        else:
            binary_array.append((0, 1))

    # print("binary array: ", binary_array)
    return x_te, binary_array

def load_HDBSCAN(x_tr):
    with open("../clusters/{}/PLELog/HDBscan_result_{}_per{}.pkl".format(dataset, ws, percent), 'rb') as f:
        list = pickle.load(f)
        # index, label, predicts, outlier, ifAnomaly, ifLabeled, confidence

    dict = {} # p(normal)
    print(list[0])
    for i in range(len(list)):
        index = list[i][0]
        conf = list[i][6]
        if (list[i][5] == 'labeled'):
            dict[index] = 1
        elif (list[i][2] == 'Normal'):
            dict[index] = 1 - conf * 0.5
        elif (list[i][2] == 'Anomalous'):
            dict[index] = conf * 0.5

    prob_array = []

    for i in range(len(x_tr)):
        prob_array.append((dict.get(i), 1 - dict.get(i)))

    return x_tr, prob_array

if __name__ =="__main__":
    gpu_available = tf.test.is_gpu_available()
    print(gpu_available)
    embed_dim = 768  # Embedding size for each token
    num_heads = 12  # Number of attention heads
    ff_dim = 2048  # Hidden layer size in feed forward network inside transformer
    max_len = 100 #75
    num_layers = 1
    dataset = 'BGL'
    ws = 'ws=20'
    k1 = 14
    k2 = 46
    dim = 50
    perform = 'Ours'
    percent = 8 # the undersampling ratio of normal and abnormal
    epo = 20

    weight_file = "../results_example/{}/{}_p={}_epoch{}.hdf5".format(dataset, ws, percent, epo)

    with open("../data/embedding/BGL/iforest-train-ws20-per{}.pkl".format(percent), mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f) # y_tr should not be used



    if perform == 'Ours':
        x_embedding, y_array = load_prob(x_tr)
        print("shape of x_embedding: ",np.shape(x_embedding))
        print("shape of y_array: ", np.shape(y_array))

    elif perform == 'HDBSCAN':
        x_embedding, y_array = load_HDBSCAN(x_tr)
        print("shape of x_embedding: ", np.shape(x_embedding))
        print("shape of y_array: ", np.shape(y_array))

    with open("../data/embedding/{}/iforest-test-ws20.pkl".format(dataset), mode="rb") as f:
        (x_te, y_te) = pickle.load(f)

    x_test, y_test = load_test(x_te, y_te)
    print(Counter(y_tr))
    print("Data loaded")

    # train(x_embedding, y_array, epo, 64, weight_file)
    print("y_test: ",len(y_test))
    F1 = test_model(x_test, y_test, batch_size=64, epochs=epo)
