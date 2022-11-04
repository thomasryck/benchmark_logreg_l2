from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    import os
    import scipy
    from download import download
    import numpy as np
    from appdirs import user_cache_dir


class Dataset(BaseDataset):
    name = "criteo"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip:appdirs']

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            cachedir = user_cache_dir("Cyanure") 
            path_X = download("http://pascal.inrialpes.fr/data2/mairal/data/criteo_X.npz", os.path.join(cachedir, "criteo_X.npz"))
            path_y = download("http://pascal.inrialpes.fr/data2/mairal/data/criteo_y.npz", os.path.join(cachedir, "criteo_y.npz"))
            dataY=np.load(os.path.join(path_y), allow_pickle=True)
            y=dataY['arr_0']
            self.X = scipy.sparse.load_npz(os.path.join(path_X))
            self.y=np.squeeze(y)

        data = dict(X=self.X, y=self.y)

        return data
