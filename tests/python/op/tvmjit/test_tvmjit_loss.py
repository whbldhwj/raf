import random
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import mnm


def get_ctx_list():
    ret = ["cpu"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret


def randn(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    t_x = torch.tensor(n_x, requires_grad=True) # pylint: disable=not-callable
    return m_x, t_x


def check(m_x, t_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    t_x = t_x.cpu().detach().numpy()
    np.testing.assert_allclose(m_x, t_x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("c", [2, 4, 6])
def test_nll_loss(ctx, n, c):
    m_pred, t_pred = randn((n, c), ctx=ctx)
    true = [random.randint(0, c - 1) for i in range(n)]
    t_true = torch.zeros([n], dtype=torch.long) # pylint: disable=no-member
    for i, j in enumerate(true):
        t_true[i] = j
    t_loss = F.nll_loss(t_pred, t_true)
    t_loss.backward()

    n_tmp = np.zeros((n, c)).astype('float32')
    for i, j in enumerate(true):
        n_tmp[i, j] = 1.0
    m_true = mnm.array(n_tmp, ctx=ctx)
    m_loss = mnm.nll_loss(m_true, m_pred)

    assert abs(m_loss.asnumpy() - t_loss.detach().numpy()) < 1e-4
    m_dpred = mnm.nll_loss_dpred(m_true, m_pred)
    np.testing.assert_allclose(m_dpred.asnumpy(), t_pred.grad.detach().numpy(), 1e-4, 1e-4)


@pytest.mark.parametrize("shape", [
    (3, 16, 128, 128),
    (3, 16),
])
@pytest.mark.parametrize("ctx", get_ctx_list())
def test_broadcast_add(shape, ctx):
    m_a, t_a = randn(shape, ctx=ctx)
    n = len(shape)
    m_b, t_b = randn([shape[1]] + [1] * (n - 2), ctx=ctx)
    t_y = t_a + t_b
    m_y = mnm.add(m_a, m_b)
    check(m_y, t_y)
    m_dy, t_dy = randn(m_y.shape, ctx=ctx)
    t_y.backward(t_dy)

    def mnm_add_dx(m_dy, m_x):
        axis = mnm.get_reduce_axis(m_dy, m_x)
        keep = mnm.get_kept_dims(m_dy, m_x)
        return mnm.sum(m_dy, axis, keep)

    m_da = mnm_add_dx(m_dy, m_a)
    check(m_da, t_a.grad, atol=0.01 if n == 4 else 1e-4)
    m_db = mnm_add_dx(m_dy, m_b)
    check(m_db, t_b.grad, atol=0.01 if n == 4 else 1e-4)


@pytest.mark.parametrize("shape", [[4, 4]])
def test_sgd(shape):
    x0 = np.random.randn(*shape).astype('float32')
    dx = np.random.randn(*shape).astype('float32')
    v0 = np.random.randn(*shape).astype('float32')
    mu = np.random.randn(1)[0] # pylint: disable=invalid-name
    learning_rate = 0.01

    n_v1 = (mu * v0 + dx)
    n_x1 = (x0 - learning_rate * n_v1)

    x0 = mnm.array(x0)
    dx = mnm.array(dx)
    v0 = mnm.array(v0)
    m_v1, m_x1 = mnm.sgd(x0, dx, v0, learning_rate, mu)

    np.testing.assert_allclose(m_v1.asnumpy(), n_v1, 1e-4, 1e-4)
    np.testing.assert_allclose(m_x1.asnumpy(), n_x1, 1e-4, 1e-4)


if __name__ == "__main__":
    pytest.main([__file__])