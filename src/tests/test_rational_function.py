import unittest
# TODO: import the unit test for the rational function python thing


class TestRationalFunctions(unittest.TestCase):

    # def test_constant_function(self) -> None:
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_linear_function(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    def test_quadratic_function(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    # def test_inverse_monomial_function(self) -> None:
    #     pass

    # def test_inverse_quadratic_function(self) -> None:
    #     pass

    # def test_rational_function(self) -> None:
    #     pass

    # def test_planar_rational_function(self) -> None:
    #     pass


if __name__ == '__main__':
    unittest.main()
