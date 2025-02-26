from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import scipy
    import numpy as np
    from cyanure import estimators
    import warnings
    from sklearn.exceptions import ConvergenceWarning

# 'solver': ['catalyst-miso', 'qning-miso', 'qning-ista',  'auto',  'acc-svrg']
class Solver(BaseSolver):
    name = 'cyanure'

    install_cmd = 'conda'
    requirements = ['cyanure']

    parameters = {
        'solver': ['catalyst-miso', 'qning-miso', 'qning-ista',  'auto',  'acc-svrg']
    }

    def set_objective(self, X, y, lmbd, name):
        self.X, self.y, self.lmbd = X, y, lmbd
        
        if (scipy.sparse.issparse(self.X) and
                scipy.sparse.isspmatrix_csc(self.X)):
            self.X = scipy.sparse.csr_matrix(self.X)

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.solver_parameter = dict(
        lambda_1= self.lmbd / X.shape[0], duality_gap_interval=10000000,
        tol=1e-15, verbose=False, solver=self.solver, max_iter=500
        )

        self.solver_instance = estimators.Classifier(loss='logistic', penalty='l2',
                                       fit_intercept=False,
                        **self.solver_parameter)

        self.dataset = "New dataset"

        self.y[self.y==0] = -1

    def run(self, n_iter):
        self.solver_instance.max_iter = n_iter
        self.solver_instance.fit(self.X, self.y)

    def get_result(self):
        return np.squeeze(self.solver_instance.get_weights())
