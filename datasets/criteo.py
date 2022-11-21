import benchopt
from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    import os
    import scipy
    from download import download
    import numpy as np


class Dataset(BaseDataset):
    name = "criteo"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip:appdirs']

    root_url = "http://pascal.inrialpes.fr/data2/cyanure/datasets"
    x_url = root_url + "/criteo_X.npz"
    y_url = root_url + "/criteo_y.npz"

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            root_path = os.path.dirname(benchopt.__file__)
            cachedir = root_path + os.path.sep + "cache"
            path_X = download(self.x_url, 
                              os.path.join(cachedir, "criteo_X.npz"))
            path_y = download(self.y_url, 
                              os.path.join(cachedir, "criteo_y.npz"))
            dataY = np.load(os.path.join(path_y), allow_pickle=True)
            y = dataY['arr_0']
            self.X = scipy.sparse.load_npz(os.path.join(path_X))
            self.y = np.squeeze(y)

        data = dict(X=self.X, y=self.y)

        return data
