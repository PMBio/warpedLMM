import numpy as np

_exp_lim_val = np.finfo(np.float64).max
_lim_val = 36.0
epsilon = np.finfo(np.float64).resolution

class Exponent(object):
    def f(self, x):
        return np.where(x<_lim_val, np.where(x>-_lim_val, np.exp(x), np.exp(-_lim_val)), np.exp(_lim_val))
    def finv(self, x):
        return np.log(x)
    def gradfactor(self, f):
        return f
    def initialize(self, f):
        return np.abs(f)    
    def __str__(self):
        return '+ve'

class Linear(object):
    def f(self, x):
        return x
    def finv(self, x):
        return x
    def gradfactor(self, f):
        return 1.0
    def initialize(self, f):
        return f
    def __str__(self):
        return ''

class Logexp(object):
    def f(self, x):
        return np.where(x>_lim_val, x, np.log(1. + np.exp(np.clip(x, -_lim_val, _lim_val)))) + epsilon
        #raises overflow warning: return np.where(x>_lim_val, x, np.log(1. + np.exp(x)))
    def finv(self, f):
        return np.where(f>_lim_val, f, np.log(np.exp(f+1e-20) - 1.))
    def gradfactor(self, f):
        return np.where(f>_lim_val, 1., 1. - np.exp(-f))
    def initialize(self, f):
        return np.abs(f)
    def __str__(self):
        return '+ve'

# class logexp(transformation):
#     domain = POSITIVE
#     def f(self, x):
#         return np.log(1. + np.exp(x))
#     def finv(self, f):
#         return np.log(np.exp(f) - 1.)
#     def gradfactor(self, f):
#         ef = np.exp(f)
#         return (ef - 1.) / ef
#     def initialize(self, f):
#         return np.abs(f)
#     def __str__(self):
#         return '(+ve)'
