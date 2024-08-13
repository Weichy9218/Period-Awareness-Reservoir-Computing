import numpy as np
import torch
import random
import time
import os
from PerioRes import PerioRes
from extract_period import extract_period
import torch.utils.data as Data
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

random_seed = 0
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def get_data(data_path, dataset, input_scaling=1.0):
    path = data_path + dataset
    data_dict = np.load(path)
    train_x = torch.from_numpy(data_dict['train_x'].astype(np.float32))
    train_y = torch.from_numpy(data_dict['train_y'].astype(np.float32))
    test_x = torch.from_numpy(data_dict['test_x'].astype(np.float32))
    test_y = torch.from_numpy(data_dict['test_y'].astype(np.float32))

    train_x = input_scaling * normalize(train_x)
    test_x = input_scaling * normalize(test_x)

    num = train_x.shape[0]
    random_index = list(range(num))
    random.shuffle(random_index)
    train_x = train_x[random_index, :, :]
    train_y = train_y[random_index]
    return train_x, train_y, test_x, test_y


def normalize(data):
    """
    data: [num_of_data, time_len, input_size]
    """
    mean = torch.mean(data, dim=1)
    max_val = torch.max(torch.abs(data), dim=1)[0]
    normalized = torch.zeros(data.shape)
    for i in range(data.size(0)):
        normalized[i, :, :] = (data[i, :, :] - mean[i, :])/max_val[i, :]
    return normalized

    # mean = torch.mean(data)
    # max_val = torch.max(torch.abs(data))
    # normalized = (data - mean) / max_val
    # return normalized


def run(data_path, batch_size=64, hidden_dim=10, scaling=1.0, radius=(0.8, 0.8), k=1, decompose="wavelet",
        wave="db4", method="periodogram", mode="original", classifier_name="rf"):
    dataset_list = os.listdir(data_path)
    results = {}
    results_path = "../results/%s_%s_%s_%s_%s_seed-%d_k-%d_batch-%d_hidden-%d_scale-%.1f_radii-%.2f-%.2f.npy" % \
                   (mode, decompose, wave, method, classifier_name, random_seed, k, batch_size, hidden_dim, scaling,
                    radius[0], radius[1])
    start = time.time()
    for dataset in dataset_list[:]:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        data_name = dataset.split('.')[0]
        results[data_name] = 0

        x_train, y_train, x_test, y_test = get_data(data_path, dataset, scaling)

        avg_x_train = torch.mean(x_train, dim=(0, 2)).unsqueeze(dim=1).numpy()
        periods = extract_period(avg_x_train, k, decompose, wave, method, mode)[0]

        tr_loader = Data.DataLoader(dataset=Data.TensorDataset(x_train, y_train),
                                    batch_size=batch_size, shuffle=False, num_workers=2)
        te_loader = Data.DataLoader(dataset=Data.TensorDataset(x_test, y_test),
                                    batch_size=batch_size, shuffle=False, num_workers=2)
        transform = PerioRes(x_train.shape[2], periods, hidden_dim=hidden_dim, spectral_radius=radius)

        train_features = []
        for x, _ in tr_loader:
            train_features.append(transform(x))
        train_features = np.concatenate(train_features, axis=0)

        test_features = []
        for x, _ in te_loader:
            test_features.append(transform(x))
        test_features = np.concatenate(test_features, axis=0)

        y_train = y_train.numpy()
        y_test = y_test.numpy()

        if classifier_name in ["rf", "max"]:
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(train_features, y_train)
            accuracy = rf.score(test_features, y_test)
            results[data_name] = max(results[data_name], accuracy)

        if classifier_name in ["ridge", "max"]:
            # Ridge
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier.fit(train_features, y_train)
            accuracy = classifier.score(test_features, y_test)
            results[data_name] = max(results[data_name], accuracy)

        if classifier_name in ["knn", "max"]:
            # K-Nearest Neighbors
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(train_features, y_train)
            accuracy = knn.score(test_features, y_test)
            results[data_name] = max(results[data_name], accuracy)

        if classifier_name in ["svm", "max"]:
            # Support Vector Machine
            svm = SVC(kernel='linear')  # 使用线性核
            svm.fit(train_features, y_train)
            accuracy = svm.score(test_features, y_test)
            results[data_name] = max(results[data_name], accuracy)

        if classifier_name in ["lda", "max"]:
            # Linear Discriminant Analysis
            lda = LinearDiscriminantAnalysis()
            lda.fit(train_features, y_train)
            accuracy = lda.score(test_features, y_test)
            results[data_name] = max(results[data_name], accuracy)

        if classifier_name in ["lr", "max"]:
            # Logistic Regression
            logreg = LogisticRegression(solver='lbfgs', max_iter=300)
            logreg.fit(train_features, y_train)
            accuracy = logreg.score(test_features, y_test)
            results[data_name] = max(results[data_name], accuracy)

        np.save(results_path, results)
        print('数据集:%s    分类准确率是:' % data_name, results[data_name])
    end = time.time()

    print("Average accuracy: %.4f" % (sum(list(results.values()))/len(results)))
    results["total_time"] = end - start
    np.save(results_path, results)
    print("-------------------------------\n总耗时:%.2fs" % (end - start))
    return 0


if __name__ == "__main__":
    run("../data/UCR/", scaling=1)
    # for s in [1, 5, 10, 20, 40, 60, 80, 100, 150, 200]:
    #     run("../data/UCR/", scaling=s)

