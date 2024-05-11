import sys
from unittest import TestCase

from numpy.testing import assert_equal


class Evaluate(TestCase):
    def test_hello_pi(self):
        import exercise  # Imports and runs student's solution
        output = sys.stdout.getvalue()  # Returns output since this function started
        assert_equal(output, 'Hello  3.141592653589793\n')
