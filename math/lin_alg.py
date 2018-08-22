import numpy as np
from fractions import Fraction

# Basis vectors


def calculateFraction(val):
    return Fraction(val).limit_denominator(10000)

def basis_on_new_vector(vec, b1, b2):
    v = np.array(vec)
    vb1 = (np.dot(v, b1)/np.dot(b1, b1))
    vb2 = (np.dot(v, b2)/np.dot(b2, b2))
    return (calculateFraction(vb1), calculateFraction(vb2))

