import functools
from collections.abc import Iterable
from math import ceil, sqrt
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm, trange

backprop = {}

float_t = np.float32

eps = np.finfo(float_t).eps


def _to_np(tensors: Iterable[Any]) -> Iterable[np.ndarray]:
    return tuple(x.view(np.ndarray) if isinstance(x, Variable) else x for x in tensors)


def _build_graph(op, args, arr, kwargs):
    if arr is NotImplemented:
        return NotImplemented
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    arr = arr.view(Variable)
    if Variable.requires_grad:
        arr.op = op
        arr.parents = args
        arr.kwargs = kwargs
    return arr


class Variable(np.ndarray):
    requires_grad = True

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.op = getattr(obj, "op", None)
        self.parents = getattr(obj, "parents", None)
        self.kwargs = getattr(obj, "kwargs", None)
        self.grad = np.zeros(obj.shape)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        np_args = _to_np(args)
        if "out" in kwargs:
            kwargs["out"] = _to_np(kwargs["out"])
        arr = super().__array_ufunc__(ufunc, method, *np_args, **kwargs)
        return _build_graph(ufunc, args, arr, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        np_args = _to_np(args)
        arr = func(*np_args, **kwargs)
        return _build_graph(func, args, arr, kwargs)

    def backward(self, grad=None):
        if grad is None:
            # Is root (ie loss node)
            grad = np.ones(self.shape)

        if (
            len(grad.shape) == len(self.grad.shape) + 1
            # and grad.shape[1:] == self.grad.shape
        ):
            # Average gradient in minibatch
            # print("minibatch", grad.shape)
            grad = np.mean(grad, axis=0)

        # print(self.op, self.grad.shape, grad.shape)
        # Accumulate gradient
        # (instead of simply setting it to deal with nodes with multiple children)
        grad = np.broadcast_to(grad, self.grad.shape)
        self.grad += grad

        for i, parent in enumerate(self.parents or []):
            if isinstance(parent, Variable):
                backprop_grad = backprop[self.op](self.parents, i, grad, **self.kwargs)
                parent.backward(backprop_grad)

    def np(self):
        return self.view(np.ndarray)

    def clip_grad_norm_(self, max_norm: int = 1):
        norm = np.linalg.norm(self.grad)
        clip_coef = max_norm / (norm + eps)
        if clip_coef < 1:
            self.grad *= clip_coef


class no_grad:
    def __enter__(self):
        self.prev = Variable.requires_grad
        Variable.requires_grad = False

    def __exit__(self, *args):
        Variable.requires_grad = self.prev


def custom_op(func):
    """Bypass numpy hooks so we can implement our own backprop logic"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if any(isinstance(arg, Variable) for arg in args):
            np_args = _to_np(args)
            res = func(*np_args, **kwargs)
            if isinstance(res, tuple):
                arr, cache = res
            else:
                arr, cache = res, {}
            return _build_graph(wrapper, args, arr, {**kwargs, **cache})
        else:
            return func(*args, **kwargs)

    return wrapper


@custom_op
def flatten(x):
    return x.reshape((x.shape[0], -1))


@custom_op
def conv2d(kernel, x, padding: int = 0):
    """
    kernel : (C_out, C_in, K, K)
    x : (B, C_in, H, W)
    out :
        "valid" : (B, C_out, H - K + 1, W - K + 1)
        "same" : (B, C_out, H, W)
        "full" : (B, C_out, H + K - 1, W + K - 1)
    """
    if padding > 0:
        x = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
    kernel_shape = kernel.shape[-2:]
    cout, cin, *_ = kernel.shape
    b, _cin, h, w = x.shape
    # sub_tensors : (B, C_in, N-K+1, N-K+1, K, K)
    sub_tensors = np.lib.stride_tricks.as_strided(
        x,
        shape=(b, 1, cin)
        + tuple(np.subtract(x.shape[-2:], kernel_shape) + 1)
        + kernel_shape,
        strides=(x.strides[0], x.strides[0], x.strides[1])
        + x.strides[-2:]
        + x.strides[-2:],
    )
    kernel_broadcast = np.expand_dims(kernel, axis=(0, 3, 4))
    return np.sum(sub_tensors * kernel_broadcast, axis=(2, -1, -2))


@custom_op
def max_pool2d(x, kernel_size: int, stride: int | None = None, padding: int = 0):
    """
    x : (B, C, H, W)
    y : (B, C, H_out, W_out)
    """

    if padding > 0:
        x = np.pad(
            x,
            [(0, 0), (0, 0), (padding, padding), (padding, padding)],
            constant_values=-np.inf,
        )
    if stride is None:
        stride = kernel_size
    b, c, h, w = x.shape
    height = ((h - kernel_size) // stride) + 1
    width = ((w - kernel_size) // stride) + 1
    sub_tensors = np.lib.stride_tricks.as_strided(
        x,
        shape=(b, c, height, width, kernel_size, kernel_size),
        strides=(
            x.strides[0],
            x.strides[1],
            x.strides[2] * stride,
            x.strides[3] * stride,
            x.strides[2],
            x.strides[3],
        ),
    )
    y = np.max(sub_tensors, axis=(-1, -2))
    y_broadcast = np.tile(y[..., None, None], (1, 1, 1, 1, kernel_size, kernel_size))
    mask = np.isclose(sub_tensors, y_broadcast)
    return y, {"mask": mask, "x_strides": x.strides}


def binary_accuracy(y_pred, y):
    return np.mean((y_pred >= 0.5) == y)


def sigmoid(x):
    return np.piecewise(
        x,
        [x > 0],
        [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
    )


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    # Stable implementation
    z = x - np.max(x, axis=-1)[:, np.newaxis]
    numerator = np.exp(z)
    return numerator / np.sum(numerator, axis=-1)[:, np.newaxis]


@custom_op
def log_softmax(x):
    # Stable implementation
    shiftx = x - np.max(x, axis=-1)[:, np.newaxis]
    return shiftx - np.log(np.sum(np.exp(shiftx), axis=-1))[:, np.newaxis]


@custom_op
def dropout(x, *, mask, scale):
    return x * mask * scale


def mse_loss(y_pred, y):
    return np.mean(np.square(y - y_pred))


@custom_op
def bce_with_logits_loss(y_pred, y):
    z = ((1 - y) * y_pred) - np.log(sigmoid(y_pred) + eps)
    return np.atleast_1d(np.mean(z))


@custom_op
def nll_loss(y_pred, y):
    """y_pred is one-hot encoded (batch, logits), y is class labels (batch,)"""
    return -np.mean(y_pred[range(y.shape[0]), y])


def bprop(forward_fn):
    def decorator(backward_fn):
        @functools.wraps(backward_fn)
        def wrapper(inputs, idx, G, **kwargs):
            return backward_fn(_to_np(inputs), idx, G, **kwargs)

        backprop[forward_fn] = wrapper
        return wrapper

    return decorator


@bprop(np.add)
def bprop_add(inputs, idx, G, **kwargs):
    x = inputs[0] if idx == 0 else inputs[1]
    if x.shape == G.shape:
        return G
    padded_shape = (1,) * (len(G.shape) - len(x.shape)) + x.shape
    sum_axes = tuple(
        i for i, (Gi, xi) in enumerate(zip(G.shape, padded_shape)) if Gi > 1 and xi == 1
    )
    return np.sum(G, axis=sum_axes)


@bprop(np.subtract)
def bprop_subtract(inputs, idx, G, **kwargs):
    return G if idx == 0 else -G


@bprop(np.square)
def bprop_square(inputs, idx, G, **kwargs):
    return 2.0 * inputs[0] * G


@bprop(np.mean)
def bprop_mean(inputs, idx, G, **kwargs):
    return np.atleast_1d(G / inputs[0].size)


@bprop(np.matmul)
def bprop_matmul(inputs, idx, G, **kwargs):
    A, B = inputs
    match idx:
        case 0:
            return G @ np.moveaxis(B, -1, -2)
        case 1:
            return A.T @ G


@bprop(np.maximum)
def bprop_maximum(inputs, idx, G, **kwargs):
    A, B = inputs
    mask = A > B if idx == 0 else B > A
    return mask.astype(float_t) * G


@bprop(flatten)
def bprop_flatten(inputs, idx, G, **kwargs):
    if G.shape == (1,):
        return np.broadcast_to(G, inputs[0].shape)
    else:
        return G.reshape(inputs[0].shape)


@bprop(conv2d)
def bprop_conv2d(inputs, idx, G, **kwargs):
    K, x = inputs
    match idx:
        case 0:
            # grad w.r.t kernel
            # The convolution between input and output gradient
            # for every combination of C_out and C_in
            # G : (B, C_out, H, W)
            # x : (B, C_in, H, W)
            # G' : (1, B, H, W)
            # x' : (C_in, B, H, W)
            b, c_out, h, w = G.shape
            c_in = x.shape[1]
            grad = np.zeros((c_out, c_in, *K.shape[2:]))
            for c in range(c_out):
                grad[c] = conv2d(
                    G[None, :, c],
                    x.transpose(1, 0, 2, 3),
                    kwargs["padding"],
                ).squeeze(1)
            return grad
        case 1:
            # grad w.r.t input
            # The full convolution between the kernel and output gradient rotated 180 deg

            # K: (C_out, C_in, K, K)
            # G: (B, C_out, H_out, W_out)
            # g: (B, C_in, H, W)

            if kwargs["padding"] == 0:
                padding = K.shape[-1] - 1
            elif kwargs["padding"] == (K.shape[-1] - 1) // 2:
                padding = kwargs["padding"]
            else:
                raise Exception("Unsupported")
            g = np.rot90(
                conv2d(
                    np.transpose(K, axes=(1, 0, 2, 3)),
                    np.rot90(G, k=2, axes=(-1, -2)),
                    padding=padding,
                ),
                k=2,
                axes=(2, 3),
            )
            return g


@bprop(max_pool2d)
def bprop_max_pool2d(inputs, idx, G, **kwargs):
    kernel_size = kwargs["kernel_size"]
    stride = kwargs.get("stride", kernel_size)
    padding = kwargs.get("padding", 0)
    x_strides = kwargs["x_strides"]
    mask = kwargs["mask"]

    x = inputs[0]
    b, c, h, w = x.shape

    # Initialize gradient array with padding
    g = np.zeros((b, c, h + 2 * padding, w + 2 * padding))

    # Get output dimensions
    out_h = ((h + 2 * padding - kernel_size) // stride) + 1
    out_w = ((w + 2 * padding - kernel_size) // stride) + 1

    # Iterate through each position in the output gradient
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride

            # Use the mask to distribute gradients only to max positions
            window_mask = mask[:, :, i, j, :, :]
            grad_window = g[
                :, :, h_start : h_start + kernel_size, w_start : w_start + kernel_size
            ]
            grad_window += G[:, :, i, j, None, None] * window_mask

    # Remove padding if necessary
    if padding > 0:
        g = g[..., padding:-padding, padding:-padding]

    return g


@bprop(np.squeeze)
def bprop_squeeze(inputs, idx, G, **kwargs):
    return np.expand_dims(G, -1)


@bprop(np.expand_dims)
def bprop_expand_dims(inputs, idx, G, **kwargs):
    return np.squeeze(G, -1)


@bprop(log_softmax)
def bprop_log_softmax(inputs, idx, G, **kwargs):
    return G - softmax(inputs[0]) * G.sum(axis=1, keepdims=True)


@bprop(dropout)
def bprop_dropout(inputs, idx, G, *, mask, scale):
    return G * mask * scale


@bprop(bce_with_logits_loss)
def bprop_bce_with_logits_loss(inputs, idx, G, **kwargs):
    """Lifted from PyTorch
    https://github.com/pytorch/pytorch/blob/e30c55ee527b40d67555464b9e402b4b7ce03737/torch/csrc/autograd/FunctionsManual.cpp#L2375
    """
    y_pred, y = inputs
    z = sigmoid(y_pred)
    G = G / y_pred.shape[0]
    return -(y * (1 - z) - (1 - y) * z) * G


@bprop(nll_loss)
def bprop_nll_loss(inputs, idx, G, **kwargs):
    y_pred, y = inputs
    if idx == 0:
        G = G / y_pred.shape[0]
        return (softmax(y_pred) - np.eye(y_pred.shape[1])[y]) * G


def clip_grad_norm_(params):
    for var in params:
        var.clip_grad_norm_()


class Module:
    def parameters(self):
        out = []
        for attr in self.__dict__.values():
            if isinstance(attr, Variable):
                out.append(attr)
            elif isinstance(attr, Module):
                out.extend(attr.parameters())
        return out


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int):
        self.W = np.random.randn(out_dim, in_dim).view(Variable)
        self.b = np.random.randn(out_dim).view(Variable)

    def __call__(self, x):
        return np.squeeze(self.W @ np.expand_dims(x, axis=-1), axis=-1) + self.b


class Conv2d(Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0
    ):
        self.padding = padding
        # he initialization
        n = in_channels * kernel_size * kernel_size
        sd = sqrt(2 / n)
        self.K = (
            np.random.normal(
                loc=0,
                scale=sd,
                size=(out_channels, in_channels, kernel_size, kernel_size),
            )
            .astype(float_t)
            .view(Variable)
        )
        # self.b = np.zeros((out_channels, 1, 1)).view(Variable)

    def __call__(self, x):
        x = conv2d(self.K, x, padding=self.padding)
        return x
        # return x + self.b
        # return conv2d(self.K, x, padding=self.padding) + self.b


class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p
        self.scale = 1 / (1 - p)

    def __call__(self, x):
        if Variable.requires_grad:
            mask = np.random.binomial(1, p=1 - self.p, size=x.shape)
            return dropout(x, mask=mask, scale=self.scale)
        return x


class Optimizer:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for var in self.params:
            var.grad[...] = 0.0


class SGD(Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0, momentum=0):
        """
        lr : learning rate
        weight_decay : L2 penalty
        """
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.v = 0

    def step(self):
        with no_grad():
            for i, var in enumerate(self.params):
                step = -self.lr * var.grad
                if self.weight_decay > 0:
                    var *= 1 - self.lr * self.weight_decay
                if self.momentum > 0:
                    step = step * self.momentum + step
                var += step


class AdamW(Optimizer):
    """https://arxiv.org/pdf/1711.05101"""

    def __init__(
        self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros(param.shape) for param in params]
        self.v = [np.zeros(param.shape) for param in params]
        self.t = 1

    def step(self):
        with no_grad():
            for i, var in enumerate(self.params):
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * var.grad
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(var.grad)
                m_hat = self.m[i] / (1 - self.b1**self.t)
                v_hat = self.v[i] / (1 - self.b2**self.t)
                var -= (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)
                var *= 1 - self.lr * self.weight_decay
        self.t += 1


def train(
    model,
    X_train,
    y_train,
    loss_fn,
    metrics_fn,
    optim,
    X_val=None,
    y_val=None,
    batch_size=64,
    max_epochs=100,
    log_epochs=1,
    log_iters=1,
    early_stopping_delta=1e-6,
    early_stopping_patience=4,
):
    n_split = ceil(X_train.shape[0] / batch_size)
    n_val_split = ceil(X_val.shape[0] / batch_size)
    X_batches, y_batches = (
        np.array_split(X_train, n_split),
        np.array_split(y_train, n_split),
    )
    x_batches = [x.view(Variable) for x in X_batches]
    y_batches = [y.view(Variable) for y in y_batches]
    x_val_batches = [x.view(Variable) for x in np.array_split(X_val, n_val_split)]
    y_val_batches = np.array_split(y_val, n_val_split)
    losses = []
    val_losses = []
    metrics = []
    val_metrics = []
    min_loss = float("inf")
    params = model.parameters()
    start = perf_counter()
    for i in (epoch_pbar := trange(max_epochs, desc="Epoch")):
        running_loss = 0.0
        running_acc = 0.0
        val_loss = val_acc = None
        if X_val is not None:
            with no_grad():
                running_val_loss = 0.0
                running_val_acc = 0.0
                for X_batch, y_batch in tqdm(zip(x_val_batches, y_val_batches), desc="Val iter", total=len(x_val_batches), leave=False):
                    y_pred = model(X_batch).view(np.ndarray)
                    running_val_loss += loss_fn(y_pred, y_batch).item()
                    running_val_acc += metrics_fn(y_pred, y_batch).item()
                val_loss = running_val_loss / len(x_val_batches)
                val_acc = running_val_acc / len(x_val_batches)
                val_losses.append(val_loss)
                val_metrics.append(val_acc)
        with tqdm(
            desc="Iter", total=len(x_batches), leave=False, disable=len(x_batches) == 1
        ) as iter_pbar:
            for j, (X_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
                optim.zero_grad()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                running_loss += loss.item()
                acc = metrics_fn(y_pred.view(np.ndarray), y_batch.view(np.ndarray))
                running_acc += acc.item()
                if j % log_iters == 0:
                    iter_pbar.set_postfix(loss=loss.item(), acc=acc.item())
                loss.backward()
                clip_grad_norm_(params)
                optim.step()
                iter_pbar.update(1)
        loss = running_loss / len(x_batches)
        losses.append(loss)
        min_loss = min(min_loss, loss)
        acc = running_acc / len(x_batches)
        metrics.append(acc)
        if i % log_epochs == 0:
            elapsed = perf_counter() - start
            epoch_pbar.set_postfix(
                loss=loss, acc=acc, val_loss=val_loss, val_acc=val_acc, elapsed=elapsed
            )
        if len(losses) >= early_stopping_patience:
            if all(
                abs(x - min_loss) < early_stopping_delta
                for x in losses[-early_stopping_patience:]
            ):
                break
    elapsed = perf_counter() - start
    print(
        f"Total epochs = {i + 1} | elapsed = {elapsed:.2f}s | train loss = {loss} | train acc = {acc}"
        f" | val loss = {val_loss} | val acc = {val_acc}"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(pd.DataFrame({"train": losses, "val": val_losses or None}), ax=ax1)
    ax1.set(xlabel="Epochs", ylabel="Loss")
    sns.lineplot(pd.DataFrame({"train": metrics, "val": val_metrics or None}), ax=ax2)
    ax2.set(xlabel="Epochs", ylabel="Accuracy")
