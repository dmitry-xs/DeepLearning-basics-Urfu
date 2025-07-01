import torch


x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

f = x**2 + y**2 + z**2 + 2*x*y*z
f.backward()

print(f"Градиент по x: {x.grad.item()}")
print(f"Градиент по y: {y.grad.item()}")
print(f"Градиент по z: {z.grad.item()}")


def analytical_gradients(x, y, z):
    """Данной функцией аналитически проверим подсчет градиентов"""
    df_dx = 2*x + 2*y*z
    df_dy = 2*y + 2*x*z
    df_dz = 2*z + 2*x*y
    return df_dx, df_dy, df_dz

a_dx, a_dy, a_dz = analytical_gradients(1, 2, 3)
print(f"\ndf/dx = {a_dx}")
print(f"df/dy = {a_dy}")
print(f"df/dz = {a_dz}")


def MSE(y_pred, y_true):
    """Данная функция считает среднеквадратичную ошибку"""
    return ((y_pred - y_true)**2).mean()


X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32) # y = 2x

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = X*w + b

mse = MSE(y_pred, y)
mse.backward()

print(f'\nГрадиент по w: {w.grad.item()}')
print(f'Градиент по b: {b.grad.item()}')


def compos_func(x: torch.Tensor):
    return torch.sin(x**2 + 1)


X2 = torch.tensor(2.0, requires_grad=True)

y2 = compos_func(X2)
y2.backward()

grad = X2.grad.item()
print(f'\nГрадиент = {grad:.4f}')

torch_autograd = torch.autograd.grad(compos_func(X2), X2)[0].item()
print(f'Автоградиент = {torch_autograd:.4f}')


