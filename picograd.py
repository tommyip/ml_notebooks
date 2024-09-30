import functools
from collections.abc import Iterable
from math import ceil
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm, trange

backprop = {}


def _to_np(tensors: Iterable[Any]) -> Iterable[np.ndarray]:
    return tuple(x.view(np.ndarray) if isinstance(x, Variable) else x for x in tensors)


def _build_graph(op, args, arr):
    if arr is NotImplemented:
        return NotImplemented
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    arr = arr.view(Variable)
    if Variable.requires_grad:
        arr.op = op
        arr.parents = args
    return arr


class Variable(np.ndarray):
    requires_grad = True

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.op = getattr(obj, "op", None)
        self.parents = getattr(obj, "parents", None)
        self.grad = np.zeros(obj.shape)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        np_args = _to_np(args)
        if "out" in kwargs:
            kwargs["out"] = _to_np(kwargs["out"])
        arr = super().__array_ufunc__(ufunc, method, *np_args, **kwargs)
        return _build_graph(ufunc, args, arr)

    def __array_function__(self, func, types, args, kwargs):
        np_args = _to_np(args)
        arr = func(*np_args, **kwargs)
        return _build_graph(func, args, arr)

    def backward(self, grad=None):
        if grad is None:
            # Is root (ie loss node)
            grad = np.ones(self.shape)

        if (
            len(grad.shape) == len(self.grad.shape) + 1
            and grad.shape[1:] == self.grad.shape
        ):
            # Average gradient in minibatch
            grad = np.mean(grad, axis=0)
        # Accumulate gradient
        # (instead of simply setting it to deal with nodes with multiple children)
        self.grad += grad

        for i, parent in enumerate(self.parents or []):
            if isinstance(parent, Variable):
                backprop_grad = backprop[self.op](self.parents, i, grad)
                parent.backward(backprop_grad)


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
        if all(isinstance(arg, Variable) for arg in args):
            np_args = _to_np(args)
            arr = func(*np_args, **kwargs)
            return _build_graph(wrapper, args, arr)
        else:
            return func(*args, **kwargs)

    return wrapper


class F:
    @classmethod
    def relu(cls, x):
        return np.maximum(x, 0)

    @classmethod
    def softmax(cls, x):
        # Stable implementation
        z = x - np.max(x, axis=-1)[:, np.newaxis]
        numerator = np.exp(z)
        return numerator / np.sum(numerator, axis=-1)[:, np.newaxis]

    @custom_op
    @classmethod
    def log_softmax(cls, x):
        # Stable implementation
        shiftx = x - np.max(x, axis=-1)[:, np.newaxis]
        return shiftx - np.log(np.sum(np.exp(shiftx), axis=-1))[:, np.newaxis]

    @classmethod
    def mse_loss(cls, y_pred, y):
        return np.mean(np.square(y - y_pred))

    @custom_op
    @classmethod
    def nll_loss(cls, y_pred, y):
        """y_pred is one-hot encoded (batch, logits), y is class labels (batch,)"""
        return -np.mean(y_pred[range(y.shape[0]), y])


def bprop(forward_fn):
    def decorator(backward_fn):
        @functools.wraps(backward_fn)
        def wrapper(inputs, idx, G):
            return backward_fn(_to_np(inputs), idx, G)

        backprop[forward_fn] = wrapper
        return wrapper

    return decorator


@bprop(np.add)
def bprop_add(inputs, idx, G):
    return G


@bprop(np.subtract)
def bprop_subtract(inputs, idx, G):
    A, B = inputs
    return G if idx == 0 else -G


@bprop(np.square)
def bprop_square(inputs, idx, G):
    return 2.0 * inputs[0] * G


@bprop(np.mean)
def bprop_mean(inputs, idx, G):
    return G / inputs[0].shape[0]


@bprop(np.matmul)
def bprop_matmul(inputs, idx, G):
    A, B = inputs
    match idx:
        case 0:
            return G @ np.moveaxis(B, -1, -2)
        case 1:
            return A.T @ G


@bprop(np.maximum)
def bprop_maximum(inputs, idx, G):
    A, B = inputs
    mask = A > B if idx == 0 else B > A
    return mask.astype(np.float64) * G


@bprop(np.squeeze)
def bprop_squeeze(inputs, idx, G):
    return np.expand_dims(G, -1)


@bprop(np.expand_dims)
def bprop_expand_dims(inputs, idx, G):
    return np.squeeze(G, -1)


@bprop(F.log_softmax)
def bprop_log_softmax(inputs, idx, G):
    return G - F.softmax(inputs[0]) * G.sum(axis=1, keepdims=True)


@bprop(F.nll_loss)
def bprop_nll_loss(inputs, idx, G):
    y_pred, y = inputs
    if idx == 0:
        G /= y_pred.shape[0]
        return (F.softmax(y_pred) - np.eye(y_pred.shape[1])[y]) * G


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


class AdamW:
    """https://arxiv.org/pdf/1711.05101"""

    def __init__(
        self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        self.params = params
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

    def zero_grad(self):
        for var in self.params:
            var.grad[...] = 0.0


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
    X_batches, y_batches = (
        np.array_split(X_train, n_split),
        np.array_split(y_train, n_split),
    )
    x_batches = [x.view(Variable) for x in X_batches]
    y_batches = [y.view(Variable) for y in y_batches]
    losses = []
    val_losses = []
    min_loss = float("inf")
    start = perf_counter()
    for i in (epoch_pbar := trange(max_epochs, desc="Epoch")):
        running_loss = 0.0
        running_acc = 0.0
        with tqdm(
            desc="Iter", total=len(x_batches), leave=False, disable=len(x_batches) == 1
        ) as iter_pbar:
            for j, (X_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
                optim.zero_grad()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                running_loss += loss.item()
                acc = metrics_fn(y_pred, y_batch)
                running_acc += acc.item()
                if j % log_iters == 0:
                    iter_pbar.set_postfix(loss=loss.item(), acc=acc.item())
                loss.backward()
                optim.step()
                iter_pbar.update(1)
        loss = running_loss / len(x_batches)
        losses.append(loss)
        min_loss = min(min_loss, loss)
        acc = running_acc / len(x_batches)
        val_loss = val_acc = None
        if X_val is not None:
            with no_grad():
                y_pred = model(X_val)
                val_loss = loss_fn(y_pred, y_val).item()
                val_losses.append(val_loss)
                val_acc = metrics_fn(y_pred, y_val).item()
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
    )
    return sns.lineplot(pd.DataFrame({"train": losses, "val": val_losses or None})).set(
        xlabel="Epochs", ylabel="Loss"
    )
