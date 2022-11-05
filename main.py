import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
import time
from sklearn.metrics import log_loss, accuracy_score
from datetime import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

RANDOM_SEED = 42


class CascadingBaseClassifier:

    def __init__(self, threshold, max_depth_list=None):
        self.max_depth_list = max_depth_list if max_depth_list else [1, 2, 4, 6, 10, 14, 18, 24, 30, 36, 48, 60, 72, 86,
                                                                     100]
        self.threshold = threshold
        self._estimators = []

    def create_classifier(self, **kwargs):
        return DecisionTreeClassifier(**kwargs)

    def fit(self, X_train, y_train, sample_data_randomly=None, sampling_from_certain=False, return_train_loss=False,
            vanish_class_handler=False):
        train_original_len = len(X_train)
        log_loss_results = []
        y_train_lables = set(np.unique(y_train))
        for depth in self.max_depth_list:
            train_len = len(X_train)
            curr_train_idxs = range(len(X_train))
            next_train_idxs = []
            clf = self.create_classifier(**dict(max_depth=depth, random_state=RANDOM_SEED))
            if sample_data_randomly:
                if int(sample_data_randomly * train_original_len) < train_len:
                    new_curr_train_idxs = random.sample(range(train_len),
                                                        int(sample_data_randomly * train_original_len))

                else:
                    new_curr_train_idxs = range(train_len)
                next_train_idxs = list(filter(lambda x: x not in new_curr_train_idxs, curr_train_idxs))
                curr_train_idxs = new_curr_train_idxs
            if vanish_class_handler:
                new_y_train_unique = set(np.unique(y_train[curr_train_idxs]))
                if y_train_lables > new_y_train_unique:
                    missing_labels = y_train_lables - new_y_train_unique
                    for label in missing_labels:
                        label_idx = np.argwhere(y_train == label).flatten()
                        num_of_chosen = math.ceil(len(label_idx) * 0.5)
                        label_idx_to_add = label_idx[:num_of_chosen]
                        next_train_idxs = list(set(next_train_idxs) - set(label_idx_to_add))
                        curr_train_idxs = list(set(list(curr_train_idxs) + list(label_idx_to_add)))
            clf.fit(X_train[curr_train_idxs], y_train[curr_train_idxs])
            certain_idx, non_certain_idx, y_probs, y_pred = self.get_proba_and_pred(X_train[curr_train_idxs], clf)
            if return_train_loss:
                log_loss_results.append(log_loss(y_train[curr_train_idxs], y_probs))
            self._estimators.append(clf)
            if len(non_certain_idx) == 0:
                break
            next_train_idxs = list(set(list(non_certain_idx) + list(next_train_idxs)) - set(certain_idx))
            curr_train_idxs = list(set(list(curr_train_idxs) + list(certain_idx)) - set(non_certain_idx))
            if sampling_from_certain:
                next_train_idxs = self.add_samples_from_certain(certain_idx, next_train_idxs, non_certain_idx,
                                                                train_original_len)
            old_X_train = X_train
            old_y_train = y_train
            X_train = X_train[next_train_idxs]
            y_train = y_train[next_train_idxs]
            if vanish_class_handler:
                new_y_train_unique = set(np.unique(y_train))
                if y_train_lables > new_y_train_unique:
                    missing_labels = y_train_lables - new_y_train_unique
                    for label in missing_labels:
                        label_idx = np.argwhere(old_y_train == label).flatten()
                        num_of_chosen = math.ceil(len(label_idx) * 0.3)
                        label_idx_to_add = label_idx[:num_of_chosen]
                        next_train_idxs = list(set(list(next_train_idxs) + list(label_idx_to_add)))
                    X_train = old_X_train[next_train_idxs]
                    y_train = old_y_train[next_train_idxs]
        if return_train_loss:
            return log_loss_results

    def add_samples_from_certain(self, certain_idx, idx_next_iter, non_certain_idx, train_original_len):
        non_certain_len = len(non_certain_idx)
        num_of_chosen = math.floor((train_original_len - non_certain_len) * 0.5)
        certain_idx_to_add = certain_idx[:num_of_chosen]
        idx_next_iter = np.concatenate((non_certain_idx, certain_idx_to_add), axis=0)
        return idx_next_iter

    def get_proba_and_pred(self, X, clf):
        y_probs = clf.predict_proba(X)
        y_pred = clf.predict(X)
        y_pred_prob = np.max(y_probs, axis=1)
        certain_pred = y_pred_prob - self.threshold >= 0
        certain_idx = np.argwhere(certain_pred).flatten()
        non_certain_idx = np.argwhere(~certain_pred).flatten()
        return certain_idx, non_certain_idx, y_probs, y_pred

    def predict_proba(self, X_test):
        final_probs_values = None
        final_preds_values = None
        for clf in self._estimators:
            certain_idx, non_certain_idx, y_probs, y_pred = self.get_proba_and_pred(X_test, clf)
            num_of_classes = y_probs.shape[1]
            if final_probs_values is None:
                final_probs_values = np.full((len(X_test), num_of_classes), np.array([-1] * num_of_classes))
                final_preds_values = np.full((len(X_test),), np.array([y_pred[0]]))
            try:
                final_probs_values[certain_idx] = y_probs[certain_idx]
                final_preds_values[certain_idx] = y_pred[certain_idx]
            except Exception as err:
                print(err)
            if len(non_certain_idx) == 0:
                break
        if len(non_certain_idx) > 0:
            final_probs_values[non_certain_idx] = y_probs[non_certain_idx]
            final_preds_values[non_certain_idx] = y_pred[non_certain_idx]
        return final_probs_values, final_preds_values


class CascadingImprovedClassifier(CascadingBaseClassifier):

    def __init__(self, threshold, max_depth_list=None):
        super().__init__(threshold=threshold,
                         max_depth_list=max_depth_list,
                         )

    def create_classifier(self, **kwargs):
        return RandomForestClassifier(**kwargs)


def load_data(target, **data_args):
    df = pd.read_csv(**data_args)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def preprocessing(name, X, y):
    if name == 'obesity':
        to_binary = {"yes": 1, "no": 0}
        X['Gender'] = X['Gender'].map({"Female": 1, "Male": 0})
        props_to_binary = ['SCC', 'family_history_with_overweight', 'FAVC', 'SMOKE']
        for prop in props_to_binary:
            X[prop] = X[prop].map(to_binary)
        freq_labels = {'no': 0, 'Sometimes': 1, 'Frequently': 2, "Always": 3}
        X['CAEC'] = X['CAEC'].map(freq_labels)
        X['CALC'] = X['CALC'].map(freq_labels)
        X['MTRANS'] = X['MTRANS'].map(
            {'Public_Transportation': 0, 'Walking': 1, 'Automobile': 2, 'Motorbike': 3, 'Bike': 4})
    if name == 'clothing_shopping':
        X['page 2 (clothing model)'] = X['page 2 (clothing model)'].apply(lambda x: int(str(ord(x[0])) + x[1:]))
        X = X.drop(columns=['session ID'])
        y = pd.qcut(y, q=5, labels=["VERY LOW", "LOW", "MID", "HIGH", "VERY HIGH"])
    elif name == 'drug_consumption':
        X = np.array(X)[:, 1:13]
    elif name == 'workers_productivity':
        X, y = workers_data_preprocessing(X, y)
    return np.array(X), np.array(y)


def workers_data_preprocessing(X, y):
    day_to_int = {"Sunday": 1, "Monday": 2, "Tuesday": 3, "Wednesday": 4,
                  "Thursday": 5, "Friday": 6, "Saturday": 7}
    department_to_int = {"sweing": 1, "finishing ": 2, "finishing": 3}
    y = pd.qcut(y, q=4, labels=["BAD", "AVERAGE", "GOOD", "EXCELLENT"])
    X['date'] = X['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').timestamp())
    X['quarter'] = X['quarter'].apply(lambda x: int(x[len(x) - 1]))
    X['day'] = X['day'].map(day_to_int)
    X['department'] = X['department'].map(department_to_int)
    X = X.sort_values(by='date')
    imp = IterativeImputer(random_state=RANDOM_SEED)
    imp.fit(X)
    X = imp.transform(X)
    return X, y


def get_datasets():
    wine_quality = dict(data_args=dict(filepath_or_buffer="wine_quality/winequality-red.csv", sep=";"),
                        name="wine_quality",
                        target="quality")
    drug_consumption = dict(data_args=dict(filepath_or_buffer="drug_consumption/drug_consumption.csv", header=None),
                            name="drug_consumption", target=31)
    poker_hand = dict(data_args=dict(filepath_or_buffer="poker_hand/poker-hand-training-true.csv", header=None),
                      name="poker_hand", target=10,
                      splitting=dict(filepath_or_buffer="poker_hand/poker-hand-testing.csv", header=None))
    hcv_egyptian = dict(data_args=dict(filepath_or_buffer="hcv_egyptian/HCV-Egy-Data.csv"),
                        name="hcv_egyptian", target='Baselinehistological staging')
    dry_beans = dict(data_args=dict(filepath_or_buffer="dry_bean/Dry_Bean_Dataset.csv"),
                     name="dry_beans",
                     target="Class")
    workers_productivity = dict(
        data_args=dict(filepath_or_buffer="worker_productivity/garments_worker_productivity.csv"),
        name="workers_productivity",
        target="actual_productivity")
    clothing_shopping = dict(
        data_args=dict(filepath_or_buffer="clothing_shopping/e-shop clothing 2008.csv", sep=";"),
        name="clothing_shopping",
        target="order")
    obesity = dict(
        data_args=dict(filepath_or_buffer="obesity/ObesityDataSet_raw_and_data_sinthetic.csv"),
        name="obesity",
        target="NObeyesdad")
    internet_firewall = dict(
        data_args=dict(filepath_or_buffer="internet_firewall/log2.csv"),
        name="internet_firewall",
        target="Action")
    avila = dict(data_args=dict(filepath_or_buffer="avila/avila-tr.txt", header=None),
                 name="avila", target=10,
                 splitting=dict(filepath_or_buffer="avila/avila-ts.txt", header=None))
    return [wine_quality, drug_consumption, poker_hand, hcv_egyptian, dry_beans, workers_productivity,
            clothing_shopping, obesity, internet_firewall, avila]


if __name__ == '__main__':
    datasets = get_datasets()
    classifiers = {'Base':CascadingBaseClassifier ,'Improved':CascadingImprovedClassifier}
    classifier = classifiers['Base']

    for dataset in datasets:
        print('Name: ', dataset['name'])
        start = time.time()
        if 'splitting' not in dataset:
            X, y = load_data(dataset['target'], **dataset['data_args'])
            X, y = preprocessing(dataset['name'], X, y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,
                                                                    random_state=RANDOM_SEED)
        else:
            X_train, y_train = preprocessing(dataset['name'], *load_data(dataset['target'], **dataset['data_args']))
            X_test, y_test = preprocessing(dataset['name'], *load_data(dataset['target'], **dataset['splitting']))

        clf = classifier(0.95)
        print(clf.fit(X_train, y_train, sample_data_randomly=0.3, return_train_loss=True, sampling_from_certain=True,
                      vanish_class_handler=True))  # TODO: report bad results with certain add (sample add!!)
        y_probs, y_pred = clf.predict_proba(X_test)
        log_loss_results = log_loss(y_test, y_probs)
        end = time.time()
        print("Log Loss: ", log_loss_results)
        print("Time: " + str(end - start) + " sec\n")
        print("*" * 50)
