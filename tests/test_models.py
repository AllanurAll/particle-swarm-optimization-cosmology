import numpy as np
from pso_cosmology.models import h_flat_lcdm, h_nonflat_lcdm, h_flat_xcdm


def test_flat_lcdm_h0_at_zero():
    assert np.isclose(h_flat_lcdm(np.array([0.0]), 70.0, 0.3)[0], 70.0)


def test_nonflat_reduces_to_flat():
    z=np.array([0.0,0.5,1.0])
    assert np.allclose(h_nonflat_lcdm(z,70,0.3,0.0),h_flat_lcdm(z,70,0.3))


def test_xcdm_w_minus_one_reduces_to_lcdm():
    z=np.array([0.0,0.5,1.0])
    assert np.allclose(h_flat_xcdm(z,70,0.3,-1.0),h_flat_lcdm(z,70,0.3))
