from classifiers import SVM, KRL
import argparse
from utils import get_sequence, get_mismatch, load_data
from Kernels import Kernel, get_gram_cross
import numpy as np
import csv
from tqdm import tqdm


def get_gram_matrix(x_tr, x_te, k, n_mismatch, n_kernel):

    dict_sequences = get_sequence(x_tr, k=k)
    embeddings_train = get_mismatch(
        dict_sequences,
        x_tr,
        k=k,
        n_mismatch=n_mismatch,
    )
    embeddings_test = get_mismatch(
        dict_sequences,
        x_te,
        k=k,
        n_mismatch=n_mismatch,
    )

    ker = Kernel(kernel="spectrum")
    gram_matrix = ker.compute_gram_matrix(embeddings_train)
    gram_test = get_gram_cross(embeddings_train, embeddings_test)
    return gram_matrix, gram_test


def predict_first_set(
    gram_train, gram_test, y_label, scale=25000, max_iter=1, lambd=0.00001
):
    gram_train, gram_test = gram_train[0], gram_test[0]
    krl = KRL(gram_m=gram_train / scale, max_iter=max_iter, lambd=lambd)
    krl.fit(np.array(y_label))
    y_predict_test = np.sign(krl.predict(gram_test / scale))
    return y_predict_test


def predict_second_set(
    gram_train, gram_test, y_label, scale=25000, max_iter=1, lambd=0.00001
):
    gram_train = gram_train[0] + gram_train[1] + gram_train[2]
    gram_test = gram_test[0] + gram_test[1] + gram_test[2]
    krl = KRL(gram_m=gram_train / scale, max_iter=max_iter, lambd=lambd)
    krl.fit(np.array(y_label))
    y_predict_test = np.sign(krl.predict(gram_test / scale))
    return y_predict_test


def predict_third_set(
    gram_train, gram_test, y_label, scale=20000, max_iter=1, lambd=0.00001
):

    gram_train = gram_train[0] + gram_train[1] + gram_train[2]
    gram_test = gram_test[0] + gram_test[1] + gram_test[2]

    krl = KRL(gram_m=gram_train / scale, max_iter=max_iter, lambd=lambd)
    krl.fit(np.array(y_label))
    y_pred_krl = krl.predict(gram_test / scale)

    clf = SVM(gram_m=gram_train)
    clf.fit(np.array(y_label))
    y_pred_svm = clf.predict(gram_test)

    y_pred = np.sign(y_pred_svm + y_pred_krl)
    return y_pred


def main(filename):
    """
    Main function for generating submissions.
    """
    y_pred_all = []
    X_train, y_train_all, X_test = load_data()
    for n in range(3):

        print(
            "############## working on dataset {} ###################".format(
                str(n + 1)
            )
        )
        # process
        y_train = 2 * np.array(y_train_all[2000 * n : 2000 * (n + 1)]) - 1

        k, n_mismatch = 13, 3
        if n != 0:
            print("Compute gram matrix for first kernel")
            gram_train_13_3, gram_test_13_3 = get_gram_matrix(
                X_train[2000 * n : 2000 * (n + 1)],
                X_test[1000 * n : 1000 * (n + 1)],
                k=k,
                n_mismatch=n_mismatch,
                n_kernel=n + 1,
            )

        k, n_mismatch = 12, 2
        if n != 0:
            print("Compute gram matrix for  second kernel ")
            gram_train_12_2, gram_test_12_2 = get_gram_matrix(
                X_train[2000 * n : 2000 * (n + 1)],
                X_test[1000 * n : 1000 * (n + 1)],
                k=k,
                n_mismatch=n_mismatch,
                n_kernel=n + 1,
            )

        print("Compute gram matrix for third kernel ")
        k, n_mismatch = 13, 2
        gram_train_13_2, gram_test_13_2 = get_gram_matrix(
            X_train[2000 * n : 2000 * (n + 1)],
            X_test[1000 * n : 1000 * (n + 1)],
            k=k,
            n_mismatch=n_mismatch,
            n_kernel=n + 1,
        )

        print("Training and generating prediction")
        if n == 0:
            train_grams = [gram_train_13_2]
            test_grams = [gram_test_13_2]
            y_pred = predict_first_set(train_grams, test_grams, y_train)
        elif n == 1:
            train_grams = [gram_train_13_2, gram_train_12_2, gram_train_13_3]
            test_grams = [gram_test_13_2, gram_test_12_2, gram_test_13_3]
            y_pred = predict_second_set(train_grams, test_grams, y_train)
        else:
            train_grams = [gram_train_13_2, gram_train_12_2, gram_train_13_3]
            test_grams = [gram_test_13_2, gram_test_12_2, gram_test_13_3]
            y_pred = predict_third_set(train_grams, test_grams, y_train)

        y_pred = (y_pred + 1) / 2
        y_pred_all += list(y_pred)

    print("Saving prediction in CSV file")

    with open(filename, "w") as csvfile:
        fieldnames = ["Id", "Bound"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in tqdm(range(0, len(y_pred_all))):
            writer.writerow({"Id": i, "Bound": int(y_pred_all[i])})

    print("You can find results on " + filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="KMML code ")
    parser.add_argument(
        "--output_file",
        type=str,
        metavar="f",
        default="kmml_preds.csv",
        help="csv output filename",
    )
    args = parser.parse_args()
    main(args.output_file)
