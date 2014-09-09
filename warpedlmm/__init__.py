from numpy.testing import Tester
from nose.tools import nottest
import testing
from stepwise import warped_stepwise as stepwise

@nottest
def tests():
    Tester(testing).test(verbose=10)
