import numpy as np
import pytest
from raf.testing.common import to_torch_dev
import torch
#import mxnet as mx
import raf

from raf.testing import (
    get_testable_devices,
    randn,
    randn_torch,
    randint,
    check,
    run_vm_model,
)


def test_unique_dim(shape, device, dtype):
    class UniqueDimModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf.unique_dim(x, dim=0, return_inverse=True, return_counts=True)

    # Skip float16 tests on CPU since it may not be supported and not much performance benefit.
    if dtype == "float16" and device == "cpu":
        pytest.skip("float16 is not supported on CPU")

    #print(shape, device, dtype)    
    m_model = UniqueDimModel()
    n_x = np.array([[4,4,3], [2,2,4], [1,3,2], [2,2,4]], np.int32)
    m_x = raf.array(n_x, device=device)
    t_x = torch.tensor(n_x, device=to_torch_dev(device))    
    #import pdb; pdb.set_trace()
    t_output, t_inverse_indices, t_counts = torch.unique(t_x, sorted=True, return_inverse=True, return_counts=True, dim=0)
    #import pdb; pdb.set_trace()
    m_output, m_inverse_indices, m_counts = m_model(m_x)
    import pdb; pdb.set_trace()
    v_output, v_inverse_indices, v_counts = run_vm_model(m_model, device, [m_x])    
    
    import pdb; pdb.set_trace()
    #check(m_output.numpy().reshape(m_output.shape[0],), t_output)
    #check(v_output.numpy().reshape(m_output.shape[0],), t_output)
    #check(m_res, t_res)
    #check(v_res, t_res)

#test_unique_dim((4,3), 'cuda', 'int32')

def test_unique_dim_single():
    device = "cuda"
    n_x = np.array([[4,4,3], [2,2,4], [1,3,2], [2,2,4]], np.int32)
    m_x = raf.array(n_x, device=device)
    #import pdb; pdb.set_trace()
    ret = raf.unique_dim(m_x, dim=0, return_inverse=True, return_counts=True)
    #import pdb; pdb.set_trace()
    #del m_x
    #del ret
    #import pdb; pdb.set_trace()
    
test_unique_dim_single()

'''
if __name__ == "__main__":
    pytest.main([__file__])
'''