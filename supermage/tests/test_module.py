"""
Tests for module in package_name.
"""

import math

import numpy as np

from .base_test import BaseTestCase, unittest


class NumbersTest(BaseTestCase):
    def test_even(self):
        """
        Test that numbers between 0 and 5 are all even.
        """
        for i in range(0, 6, 2):
            with self.subTest(i=i):
                self.assertEqual(i % 2, 0)


class TestVerbose(BaseTestCase):
    """
    Test that things are printed to stdout correctly.
    """

    def test_hello_world(self):
        """Test printing to stdout."""
        message = "Hello world!"
        capture_pre = self.capsys.readouterr()  # Clear stdout
        print(message)  # Execute method (verbose)
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assert_string_equal(capture_post.out.strip(), message)

    def test_shakespeare(self):
        # Clear stdout (in this case, an empty capture)
        capture_pre = self.capsys.readouterr()
        # Execute method (verbose)
        print("To be, or not to be, that is the question:")
        # Capture the output to stdout, then re-output
        capture_post = self.recapsys(capture_pre)
        # Compare output to target
        self.assert_starts_with(capture_post.out, "To be, or not")
        # Clear stdout (in this case, capturing the re-output first print statement)
        capture_pre = self.capsys.readouterr()
        # Execute method (verbose)
        print("Whether 'tis nobler in the mind to suffer")
        # Capture the output to stdout, then re-output. This now prints both
        # lines to stdout at once, which otherwise would not appear due to our
        # captures.
        capture_post = self.recapsys(capture_pre)
        # Compare output to target
        self.assert_starts_with(capture_post.out.lower(), "whether 'tis nobler")


class TestAlways(BaseTestCase):
    def test_example(self):
        self.assertTrue(True)
