import benchopt
from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    import os
    import numpy as np
    from download import download
    import scipy.sparse


class Dataset(BaseDataset):
    name = "webspam"
    is_sparse = True

    root_url = "http://pascal.inrialpes.fr/data2/mairal/data/"
    x_url = root_url + "/webspam_X.npz"
    y_url = root_url + "/webspam_y.npz"

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):
        if self.X is None:
            root_path = os.path.dirname(benchopt.__file__)
            print(root_path)
            cachedir = root_path + os.path.sep + "cache"
            path_X = download(self.x_url,
                              os.path.join(cachedir, "webspam_X.npz"))
            path_y = download(self.y_url,
                              os.path.join(cachedir, "webspam_y.npz"))
            self.X = scipy.sparse.load_npz(os.path.join(path_X))
            self.y = np.load(os.path.join(path_y), allow_pickle=True)
            self.y=self.y['arr_0']
            self.y = np.squeeze(self.y)

        data = dict(X=self.X, y=self.y, name=self.name)
        return data