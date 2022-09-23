from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    import os
    import numpy as np


class Dataset(BaseDataset):
    name = "dino"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            path_train = "/scratch/tryckebo/Téléchargements/feat_INAT_train.npy"
            path_test = "/scratch/tryckebo/Téléchargements/feat_INAT_test.npy"
            path_train_lab = "/scratch/tryckebo/Téléchargements/lab_INAT_train.npy"
            path_test_lab = "/scratch/tryckebo/Téléchargements/lab_INAT_test.npy"
            X_test=np.load(os.path.join(path_test), allow_pickle=True)
            self.X = np.load(os.path.join(path_train), allow_pickle=True)
            y_test=np.load(os.path.join(path_test_lab), allow_pickle=True)
            self.y = np.load(os.path.join(path_train_lab), allow_pickle=True)
            print(self.X)
            print(self.X.shape)
            print(np.unique(self.y))
            print(self.y.shape)
            self.y=np.squeeze(self.y)

        data = dict(X=self.X, y=self.y, X_test=X_test, y_test=y_test)
        data = dict(X=self.X, y=self.y)
        return data
