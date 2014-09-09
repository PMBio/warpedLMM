import numpy as np
import scipy as sp
import GPy
import transformations
from warping_functions import TanhWarpingFunction_d
np.random.seed(123)

class WarpedLMM(object):
    def __init__(self, Y, X, X_selected=None, K=None, warping_function='Snelson', warping_terms=3):
        #self.Y_untransformed = self._scale_data(Y)
        self.Y_untransformed = Y
        self.Y_untransformed.flags.writeable = False
        self.Y = self.Y_untransformed
        self.X = X
        self.X_selected = X_selected
        self.X.flags.writeable = False

        self.params = {'sigma_g': np.array([1.0]),
                       'sigma_e': np.array([1.0]),
                       'bias': np.array([0.01])}

        self.param_transformations = {'sigma_g': transformations.Logexp(),
                                      'sigma_e': transformations.Logexp(),
                                      'bias': transformations.Logexp()}


        self.num_data, self.output_dim = self.Y.shape
        self.input_dim = self.X.shape[1]

        if K is None:
            self.XXT = np.dot(X, X.T)
        else:
            self.XXT = K

        self.I = np.eye(self.num_data)
        self.ones = np.ones_like(self.XXT)

        if self.X_selected is not None:
            for i in range(self.X_selected.shape[1]):
                 self.params['sigma_d_%d' % i] = np.array([1.0])
                 self.param_transformations['sigma_d_%d' % i] = transformations.Logexp()

        self.jitter = 0.0

        self.warping_params_names = []
        if warping_function == "Snelson":
            self.warping_function = TanhWarpingFunction_d(warping_terms)

            self.params['tanh_d'] = 0.5
            self.param_transformations['tanh_d'] = transformations.Logexp()

            for i in range(warping_terms):
                self.params['tanh_a_%d' % i] = np.random.uniform(0, 1.0)
                self.params['tanh_b_%d' % i] = np.random.uniform(0, 1.0)
                self.params['tanh_c_%d' % i] = np.random.uniform(-0.5, 0.5)

                self.param_transformations['tanh_a_%d' % i ] = transformations.Logexp()
                self.param_transformations['tanh_b_%d' % i ] = transformations.Logexp()
                self.param_transformations['tanh_c_%d' % i ] = transformations.Linear()
                self.warping_params_names.extend(['tanh_a_%d' % i, 'tanh_b_%d' % i, 'tanh_c_%d' % i])
            self.warping_params_names.append('tanh_d')
            self.warping_params = np.array([self.params[p] for p in self.warping_params_names])

        self.params_ordering = [k for k in self.params]
        self._set_params(self._get_params())

    def randomize(self):
        for p in self.params_ordering:
            self.params[p] = self.param_transformations[p].initialize(np.random.randn())
        self._set_params(self._get_params())

    def _scale_data(self, Y):
        self._Ymax = Y.max()
        self._Ymin = Y.min()
        return (Y - self._Ymin) / (self._Ymax - self._Ymin) - 0.5

    def _unscale_data(self, Y):
        return (Y + 0.5) * (self._Ymax - self._Ymin) + self._Ymin

    def _get_params(self):
        return np.hstack([self.param_transformations[k].finv(self.params[k]) for k in self.params_ordering])

    def _set_params(self, x):
        for i, p in enumerate(self.params_ordering):
            self.params[p] = self.param_transformations[p].f(x[i])

        self.warping_params = np.array([self.params[p] for p in self.warping_params_names])
        self.Y = self.transform_data()
        self.YYT = np.dot(self.Y, self.Y.T)

        self.K_genetics = self.params['sigma_g'] * self.XXT

        if self.X_selected is not None:
            self.K_selected = np.zeros_like(self.K_genetics)
            for i in range(self.X_selected.shape[1]):
                Xsigma = self.X_selected[:, i:i+1] * np.sqrt(self.params['sigma_d_%d' % i])
                self.K_selected += np.dot(Xsigma, Xsigma.T)
        else:
            self.K_selected = 0.0

        self.K =  self.K_genetics + self.K_selected + self.params['sigma_e'] * self.I + self.params['bias'] * self.ones

        self.K_inv, _, _, self.log_det_K = GPy.util.linalg.pdinv(self.K) # TODO cache 1-kernel case

    def transform_data(self):
        Y = self.warping_function.f(self.Y_untransformed, self.warping_params)
        return Y

    def log_likelihood(self):
        ll = - 0.5 * self.num_data * np.log(2*np.pi) - 0.5 * self.log_det_K - 0.5 * np.dot(np.dot(self.Y.T, self.K_inv), self.Y)[0,0]
        jacobian = self.warping_function.fgrad_y(self.Y_untransformed, self.warping_params)
        return ll + np.log(jacobian).sum()

    def _warping_function_gradients(self, Kiy):
        grad_y = self.warping_function.fgrad_y(self.Y_untransformed, self.warping_params)
        grad_y_psi, grad_psi = self.warping_function.fgrad_y_psi(self.Y_untransformed, self.warping_params,
                                                                 return_covar_chain=True)
        djac_dpsi = ((1.0 / grad_y[:, :, None, None]) * grad_y_psi).sum(axis=0).sum(axis=0)
        dquad_dpsi = (Kiy[:, None, None, None] * grad_psi).sum(axis=0).sum(axis=0)

        return -dquad_dpsi + djac_dpsi

    def _log_likelihood_gradients(self):
        dL_dK = 0.5*np.dot(np.dot(self.K_inv, self.YYT), self.K_inv) - 0.5*self.K_inv

        gradients = {'sigma_e': np.trace(dL_dK),
                     'sigma_g': np.sum(dL_dK * self.XXT),
                     'bias': np.sum(dL_dK)}

        if self.X_selected is not None:
            for i in range(self.X_selected.shape[1]):
                gradients['sigma_d_%d' % i] = [np.sum(dL_dK * np.dot(self.X_selected[:, i:i+1], self.X_selected[:, i:i+1].T))] # TODO einsum

        alpha = np.dot(self.K_inv, self.Y.flatten())
        warping_grads = self._warping_function_gradients(alpha)
        warping_grads = np.append(warping_grads[:, :-1].flatten(), warping_grads[0, -1])

        for i in range(len(self.warping_params)):
            gradients[self.warping_params_names[i]] = warping_grads[i]

        return gradients

    def _f(self, x):
        self._set_params(x)
        return -self.log_likelihood()

    def _f_prime(self, x):
        self._set_params(x)
        gradients = self._log_likelihood_gradients()
        return -np.hstack([gradients[p] * self.param_transformations[p].gradfactor(self.params[p]) for p in self.params_ordering])

    def optimize(self, messages=1):
        x_opt = sp.optimize.fmin_l_bfgs_b(self._f, self._get_params(), fprime=self._f_prime, iprint=messages)
        self._set_params(x_opt[0])

    def optimize_restarts(self, num_restarts, messages=1):
        NLLs = []
        params = []
        for i in range(num_restarts):
            try:
                self.optimize(messages=0)
                params.append(self._get_params())
                NLLs.append(self._f(params[-1]))
                if messages == 1:
                    print "Optimization restart %d/%d, f: %.4f" % (i+1, num_restarts, NLLs[-1])
            except Exception:
                if messages == 1:
                    print "Optimization restart %d/%d: Failed (LinalgError)" % (i+1, num_restarts)
            self.randomize()

        self._set_params(params[np.argmin(NLLs)])

    def __str__(self):
        return self.params.__str__()

    def plot_warping(self):
        self.warping_function.plot(self.warping_params, self.Y_untransformed.min(), self.Y_untransformed.max())

    def checkgrad(self, step=1e-6):
        names = self.params_ordering
        header = ['Name', 'Ratio', 'Difference', 'Analytical', 'Numerical']
        max_names = max([len(names[i]) for i in range(len(names))] + [len(header[0])])
        float_len = 10
        cols = [max_names]
        cols.extend([max(float_len, len(header[i])) for i in range(1, len(header))])
        cols = np.array(cols) + 5
        header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
        header_string = map(lambda x: '|'.join(x), [header_string])
        separator = '-' * len(header_string[0])
        xx = self._get_params()
        gradient = self._f_prime(xx)
        print '\n'.join([header_string[0], separator])
        for i, p in enumerate(self.params_ordering):
            xx = xx.copy()
            xx[i] += step
            f1 = self._f(xx)
            xx[i] -= 2.*step
            f2 = self._f(xx)
            numerical_gradient = (f1 - f2) / (2*step)
            if np.all(gradient[i] == 0): ratio = (f1 - f2) == gradient[i]
            else: ratio = (f1 - f2) / (2 * step * gradient[i])
            difference = np.abs((f1 - f2) / 2 / step - gradient[i])
            r = '%.6f' % float(ratio)
            d = '%.6f' % float(difference)
            g = '%.6f' % gradient[i]
            ng = '%.6f' % float(numerical_gradient)
            grad_string = "{0:<{c0}}|{1:^{c1}}|{2:^{c2}}|{3:^{c3}}|{4:^{c4}}".format(p, r, d, g, ng, c0=cols[0], c1=cols[1], c2=cols[2], c3=cols[3], c4=cols[4])
            print grad_string

    # def predict(self, Xnew, which_parts='all', full_cov=False, pred_init=None):
    #     # normalize X values
    #     Xnew = (Xnew.copy() - self._Xoffset) / self._Xscale
    #     mu, var = GP._raw_predict(self, Xnew, full_cov=full_cov, which_parts=which_parts)

    #     # now push through likelihood
    #     mean, var, _025pm, _975pm = self.likelihood.predictive_values(mu, var, full_cov)

    #     if self.predict_in_warped_space:
    #         mean = self.warping_function.f_inv(mean, self.warping_params, y=pred_init)
    #         var = self.warping_function.f_inv(var, self.warping_params)

    #     if self.scale_data:
    #         mean = self._unscale_data(mean)

    #     return mean, var, _025pm, _975pm


if __name__ == '__main__':
    # import matplotlib.pylab as plt
    N = 120
    X = np.random.randn(N, 1)
    X -= X.mean()
    X /= X.std()
    Z = X * 0.8 + np.random.randn(N, 1)*0.02
    Z += np.abs(Z.min()) + 0.5
    Y = np.exp(Z)#(1/4.0)
    m = WarpedLMM(Y, X, warping_terms=2)
    # m.randomize()
    # m.optimize(messages=1)

    # plt.figure('warping vs truth')
    # plt.plot(Z, m.Y, 'og')
    # plt.plot(Z, Y, 'or')
    # # plt.figure('GP')
    # # plt.plot(X, Z, 'go')
    # # plt.plot(X, m._unscale_data(m.Y), 'ro')
    # print sp.stats.pearsonr(m.Y, Z)


    N = 120
    X = np.random.randn(N, 500)
    Z = np.dot(X, np.random.randn(500,1)*.1) + np.random.randn(N, 1)*0.1
    Z += np.abs(Z.min()) + 0.5
    # Y = np.exp(Z)#(1/4.0)
    Y = Z**1/4.
    m = WarpedLMM(Y, X, X_selected=X[:,0:5], warping_terms=1)
    m2 = WarpedLMM(Y, X,  warping_terms=1)
