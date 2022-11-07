import benchopt
from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    import os
    import numpy as np
    from download import download


class Dataset(BaseDataset):
    name = "ocr"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            cachedir = os.path.dirname(benchopt.__file__) + os.path.sep + "cache"
            path = download("http://pascal.inrialpes.fr/data2/mairal/data/ocr.npz", os.path.join(cachedir, "ocr.npz"))
            data = np.load(path)
            self.y = data['arr_1']
            self.y = np.squeeze(self.y)
            self.X = data['arr_0']

        data = dict(X=self.X, y=self.y)
        return data
