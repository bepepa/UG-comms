#! /usr/bin/env python3

# File: sources.py - sources and sinks;
#       various ways to produce or consume sequences of bits

"""
# Sources and Sinks

This module contains various functions that act as sources or sinks for bit sequences

## Sources:

* `string_source( string )`: convert a string to a sequence of bits
* `random_bit_source( N )`: generate a sequence of N bits

## Sinks:

* `string_sink( bits )`: convert a sequence of bits into a string

## Error Counter:
* `count_errors (tx, rx)`: count the number of differences between tx and rx

"""

import numpy as np

from comms.utils import byte_to_bits, bits_to_byte


def string_source(string):
    """convert a string to a vector of bits

    Inputs:
    -------
    * string - string to be converted into a bit-sequence; maybe ASCII or UTF-8

    Returns:
    --------
    Numpy vector of bits
    """
    # convert a string to a sequence of bytes
    bb = string.encode()
    Nb = len(bb)

    # allocate space
    bits = np.zeros(8 * Nb, dtype=np.uint8)

    for n in range(Nb):
        bits[8 * n : 8 * (n + 1)] = byte_to_bits(bb[n])

    return bits


def random_bit_source(N):
    """Generate a sequence of N random bits

    Inputs:
    -------
    * N - number of bits to generate

    Returns:
    --------
    Numpy vector of bits
    """
    return np.random.randint(0, 2, size=N)


def string_sink(bits):
    """convert a sequence of bits into a string

    Inputs:
    -------
    * bits - an iterable containing 0's and 1'; length must be a multiple of 8

    Returns:
    --------
    (string) the result of decoding sequence of bits
    """

    # check that number of bits is a multiple of 8
    if len(bits) % 8 != 0:
        raise ValueError(f"number of bits {len(bits)} is not divisible by 8.")

    # allocate storage
    n_bytes = len(bits) // 8
    bytes = np.zeros(n_bytes, dtype=np.uint8)

    for n in range(n_bytes):
        bytes[n] = bits_to_byte(bits[n * 8 : (n + 1) * 8])

    # decode the string (deal with unicode encoding)
    return bytes.tobytes().decode("utf-8", "replace")


def count_errors(tx, rx):
    """count the number of differences between input and output

    Inputs:
    -------
    * `tx`: transmitted symbols or bits
    * `rx`: received symbols or bits

    Returns:
    --------
    (int) - number of disagreements between tx and rx
    """
    epsilon = 1e-15
    return np.sum(np.abs(rx - tx) > epsilon)


if __name__ == "__main__":

    ## Round-trip test
    string = "Hi 😲"
    assert string == string_sink(string_source(string))

    N = 100
    rand_bits = random_bit_source(N)
    assert len(rand_bits) == N

    assert count_errors(rand_bits, rand_bits) == 0
    assert count_errors(rand_bits, -rand_bits) == N

    # all good if we get here
    print("OK")
