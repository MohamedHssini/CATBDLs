import itertools
import os
import json


from matplotlib.backends.backend_pdf import PdfPages

from StatisticFile import StatisticTable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TF_ENABLE_ONEDNN_OPTS = 'TF_ENABLE_ONEDNN_OPTS'
VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"

os.environ[VISIBLE_DEVICES] = "1"
os.environ[TF_ENABLE_ONEDNN_OPTS] = '0'
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from itertools import product
import Generator as Gen
import PickleHandler as PH

tf.compat.v1.disable_eager_execution()

pappers , labels= PH.getCiteseerData()
number_pappers = len(pappers)

number_output = 6
labels = np.eye(number_output)[labels]

epochs = PH.convert_to_correct_type("EPOCHS")
number_of_experiences = PH.convert_to_correct_type("NUMBER_OF_SAMPLES")

my_session = tf.compat.v1.Session()



class NetworkModel(keras.Sequential):

    def __init__(self, input_shape, output_size):
        super(NetworkModel, self).__init__()
        self.add(keras.Input(shape=input_shape))
        # self.add(keras.layers.Dense(units=250, activation="relu"))
        # self.add(keras.layers.Dense(units=125, activation="relu"))
        self.add(keras.layers.Dense(256, activation="relu"))
        self.add(keras.layers.Dense(128, activation="relu"))
        self.add(keras.layers.Dense(64, activation="relu"))
        self.add(keras.layers.Dense(units=output_size, activation="softmax"))


np.random.seed(0)

train_index = tf.compat.v1.placeholder(tf.int32, shape=[None])
test_index = tf.compat.v1.placeholder(tf.int32, shape=[None])
prameter_lambda = tf.compat.v1.placeholder(tf.float32, shape=())
learning_rate = tf.compat.v1.placeholder(tf.float32, shape=())


def get_loss(y_true, y_pred, index_input, type_generator):
    global prameter_lambda
    index_input = tf.cast(index_input, tf.int64)
    Generator = Gen.Gcreate.getGen(type_generator)
    t = Generator(p=prameter_lambda)
    fo = Gen.FuzzyOperator(t, type_generator)
    kb = Gen.KnowledgeBase()

    a = tf.gather(y_true, index_input)
    b = tf.gather(y_pred, index_input)

    kb.add(fo.implication(a, b))
    return kb.loss(type_generator)


def get_accuracy(y_true, y_pred):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1)),
            tf.float32
        )
    )


model = NetworkModel(np.shape(pappers)[1:], number_output)
y_pred = model(tf.constant(pappers, dtype=tf.float32))

y_true_train = tf.gather(labels, train_index)
y_pred_train = tf.gather(y_pred, train_index)

y_true_test = tf.gather(labels, test_index)
y_pred_test = tf.gather(y_pred, test_index)

tr_accuracy = get_accuracy(y_true_train, y_pred_train)
te_accuracy = get_accuracy(y_true_test, y_pred_test)


def train(s, p, lr, type_generator):
    data_tr_index, data_te_index = train_test_split(np.arange(number_pappers), test_size=s)
    loss = get_loss(tf.cast(labels, tf.float32), y_pred, train_index, type_generator)

    train_operation = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.compat.v1.set_random_seed(0)
    np.random.seed(0)
    my_session.run(tf.compat.v1.global_variables_initializer())

    val_test_acc = []
    val_train_acc = []
    dictionary = {prameter_lambda: p, learning_rate: lr, train_index: data_tr_index, test_index: data_te_index}
    for element in range(epochs):
        my_session.run(train_operation, feed_dict=dictionary)
        train_accuracy, test_accuracy = my_session.run((tr_accuracy, te_accuracy), feed_dict=dictionary)
        print("Epoch {}/{}: --- Train Accuracy: {:.5f} --- Test Accuracy: {:.5f}".
              format(element, epochs, train_accuracy, test_accuracy))
        val_test_acc.append(test_accuracy)
        val_train_acc.append(train_accuracy)

    return val_train_acc, val_test_acc


def format_number(num_str):
    num = float(num_str)
    num_formatted = f"{num:.10f}".rstrip("0")

    if num_formatted[-1] == ".":
        num_formatted += "0"

    if "." in num_formatted:
        integer_part, decimal_part = num_formatted.split(".")
        if len(decimal_part) > 2 and any(d != "0" for d in decimal_part[2:]):
            return f"{num:.2f}"
        elif len(decimal_part) > 2:
            return f"{integer_part}.{decimal_part[:5]}"

    return num_formatted
def get_last_elements(nested_list):
    return [inner_list[-1] for inner_list in nested_list if inner_list]



def main():

    type_generator_list = PH.convert_to_correct_type("TYPE_GENERATOR_LIST")


    split_list = PH.convert_to_correct_type("SPLIT_LIST")

    parameter_list_yager = PH.convert_to_correct_type("PARAMETER_LIST_YAGER")  # For Yager t-norm
    parameter_list_acal  = PH.convert_to_correct_type("PARAMETER_LIST_ACAL")  # For Acal t-norm

    learning_rate_list = PH.convert_to_correct_type("LEARNING_RATE_LIST")
    list_Yager = []
    list_Acal = []
    type_generator_index = 0
    for type_generators in product(type_generator_list):

        type_generator = type_generators[0]
        if type_generator == Gen.Yager :
            parameter_list = parameter_list_yager
        elif type_generator == Gen.Acal :
            parameter_list = parameter_list_acal

        pdf_filename = f'Test_Accuracy_{type_generator}_Tnorms.pdf'
        with PdfPages(pdf_filename) as pdf:
            accuracy_dictionary = {}
            print("start type_generator : {} ".format(str(type_generator)))
            list_Rows = []
            data_List = []
            ligne = {}
            each_Test = {}
            index = len(parameter_list)

            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.95, f"Results for  {type_generator} t-norms",
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=14, weight='bold', color='black')
            plt.axis('off')

            pdf.savefig()
            plt.clf()

            for spl in product(split_list, parameter_list, learning_rate_list):
                s, p, lr = spl
                print("start accuracy_dictionary_{}".format(str(spl)))
                TRAIN_ACCURACY_LIST = []
                TEST_ACCURACY_LIST = []

                for n_exp in range(number_of_experiences):
                    val_train_acc, val_test_acc = train(s, p, lr, type_generator)
                    TRAIN_ACCURACY_LIST.append(val_train_acc)
                    TEST_ACCURACY_LIST.append(val_test_acc)

                mean_test_acc = np.mean(get_last_elements(TEST_ACCURACY_LIST))
                std_test_acc = np.std(get_last_elements(TEST_ACCURACY_LIST))

                if "test_set" not in each_Test:
                    each_Test["test_set"] = s * 100

                ligne = {
                    "λ": str(p),
                    "Av Acc": format_number(str(mean_test_acc * 100)),
                    "Stddev": str(std_test_acc)
                }
                list_Rows.append(ligne.copy())

                index -= 1

                if index == 0:
                    each_Test["rows"] = list_Rows.copy()
                    data_List.append(each_Test.copy())

                    each_Test = {}
                    list_Rows = []
                    index = len(parameter_list)

                print("end accuracy_dictionary_{}".format(str(spl)))

                accuracy_dictionary[(s, p, lr)] = (
                    np.mean(TRAIN_ACCURACY_LIST, axis=0),
                    np.mean(TEST_ACCURACY_LIST, axis=0))
                PH.save_to_file(accuracy_dictionary, "accuracy_dictionary_{}_{}.pkl".format(
                    type_generator,  s))


            if type_generator == Gen.Yager:
                list_Yager = data_List
            elif type_generator == Gen.Acal:
                list_Acal = data_List

            accuracy_dictionary = PH.load_from_file("accuracy_dictionary_{}_{}.pkl".format(
                type_generator, s))

            for s in split_list:
                plt.figure()

                for p in parameter_list:
                    test_acc_for_p = []

                    for lr in learning_rate_list:
                        key = (s, p, lr)
                        if key in accuracy_dictionary:
                            test_acc_for_p.append(accuracy_dictionary[key][1][:epochs])  # Test accuracy

                    # Plot all `p` values for the current s
                    if test_acc_for_p:
                        for idx, acc in enumerate(test_acc_for_p):
                            plt.plot(acc, label=f"$\lambda = $ {p}")

                # plt.title(f"Test Accuracy for Split {s}")
                # plt.xlabel('Epochs')
                # plt.ylabel('Accuracy')
                plt.legend(loc='upper left', prop={'size': 7}, fancybox=True, framealpha=0.5)

                pdf.savefig()
                plt.clf()
    final_list = []
    if len(type_generator_list) == 2:
        for item in list_Acal:
            for row in item["rows"]:
                row["λ_A"] = row.pop("\u03bb")  # Rename "λ" to "λ_A" , '\u03bb'= lambda
                row["Av Acc_A"] = row.pop("Av Acc")  # Rename "Av Acc" to "Av Acc_A"
                row["Stddev_A"] = row.pop("Stddev")

        for set_1, set_2 in zip(list_Yager, list_Acal):
            test_set = set_1["test_set"]
            rows_1 = set_1["rows"]
            rows_2 = set_2["rows"]

            combined_rows = []

            for row_1, row_2 in itertools.zip_longest(rows_1, rows_2, fillvalue={"λ": "-", "Av Acc": "-", "Stddev": "-"}):
                combined_row = {
                    "λ_Yager": row_1.get("λ", "-"),
                    "Av Acc_Yager": row_1.get("Av Acc", "-"),
                    "Stddev_Yager": row_1.get("Stddev", "-"),
                    "λ_Acal": row_2.get("λ_A", "-"),
                    "Av Acc_Acal": row_2.get("Av Acc_A", "-"),
                    "Stddev_Acal": row_2.get("Stddev_A", "-"),
                }
                combined_rows.append(combined_row)

            final_list.append({"test_set": test_set, "rows": combined_rows})
    else:
        if type_generator_list[0] == 'Yager':
            type_generator_index=1
            final_list=list_Yager
        else:
            type_generator_index = 2
            final_list = list_Acal

    creator = StatisticTable(final_list, type_generator_index=type_generator_index)
    creator.generate_table_pdf()


if __name__ == "__main__":

    main()
