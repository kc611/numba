import platform
import numpy as np
from numba import types
import unittest
from numba import njit
from numba.tests.support import TestCase
from numba.experimental.remarks import remarks

_skylake_env = {
    "NUMBA_CPU_NAME": "skylake-avx512",
    "NUMBA_CPU_FEATURES": "",
}


@unittest.skipIf(platform.machine() != 'x86_64', 'x86_64 only test')
class TestVectorization(TestCase):
    """
    Tests to assert that code which should vectorize does indeed vectorize
    """

    @TestCase.run_test_in_subprocess(envvars=_skylake_env)
    def test_nditer_loop(self):
        # see https://github.com/numba/numba/issues/5033
        def do_sum(x):
            acc = 0
            for v in np.nditer(x):
                acc += v.item()
            return acc

        with remarks.RemarksInterface() as remarks_interface:
            njit((types.float64[::1],), fastmath=True)(do_sum)

        remarks_interface.generate_remarks(collect_passed_remarks=True)

        loop_vectorize_remarks = remarks_interface.get_remarks("loop-vectorize",
                                                               "Passed")
        assert len(loop_vectorize_remarks) == 1
        loop_vectorize_remark = loop_vectorize_remarks[0]

        assert isinstance(loop_vectorize_remark, remarks.Passed)
        assert loop_vectorize_remark.message == \
            'vectorized loop (vectorization width: 4, interleaved count: 1)'

    # SLP is off by default due to miscompilations, see #8705. Put this into a
    # subprocess to isolate any potential issues.
    @TestCase.run_test_in_subprocess(
        envvars={'NUMBA_SLP_VECTORIZE': '1', **_skylake_env},
    )
    def test_slp(self):
        # Sample translated from:
        # https://www.llvm.org/docs/Vectorizers.html#the-slp-vectorizer

        def foo(a1, a2, b1, b2, A):
            A[0] = a1 * (a1 + b1)
            A[1] = a2 * (a2 + b2)
            A[2] = a1 * (a1 + b1)
            A[3] = a2 * (a2 + b2)

        ty = types.float64
        with remarks.RemarksInterface() as remarks_interface:
            njit(((ty,) * 4 + (ty[::1],)), fastmath=True)(foo)

        remarks_interface.generate_remarks(collect_passed_remarks=True)
        slp_vect_remarks = remarks_interface.get_remarks("slp-vectorizer",
                                                         "Passed")

        assert len(slp_vect_remarks) == 1
        slp_vect_remark = slp_vect_remarks[0]
        assert isinstance(slp_vect_remark, remarks.Passed)
        assert slp_vect_remark.message == \
            'Stores SLP vectorized with cost -4 and with tree size 6'

    @TestCase.run_test_in_subprocess(envvars=_skylake_env)
    def test_instcombine_effect(self):
        # Without instcombine running ahead of refprune, the IR has refops that
        # are trivially prunable (same BB) but the arguments are obfuscated
        # through aliases etc. The follow case triggers this situation as the
        # typed.List has a structproxy call for computing `len` and getting the
        # base pointer for use in iteration.

        def sum_sqrt_list(lst):
            acc = 0.0
            for item in lst:
                acc += np.sqrt(item)
            return acc

        with remarks.RemarksInterface() as remarks_interface:
            njit((types.float64[::1],), fastmath=True)(sum_sqrt_list)

        remarks_interface.generate_remarks(collect_passed_remarks=True)

        loop_vectorize_remarks = remarks_interface.get_remarks("loop-vectorize",
                                                               "Passed")
        assert len(loop_vectorize_remarks) == 1
        loop_vectorize_remark = loop_vectorize_remarks[0]

        assert isinstance(loop_vectorize_remark, remarks.Passed)
        assert loop_vectorize_remark.message == \
            'vectorized loop (vectorization width: 4, interleaved count: 4)'


if __name__ == '__main__':
    unittest.main()
