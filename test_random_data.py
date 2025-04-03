from similarity_metrics import linear_cka_efficient, procrustes_pytorch, rsa_with_rdms, cca_similarity
import torch




n_samples = 1250
d = 64
x_scale = 10
y_scale = 10

X = torch.randn(n_samples, d) * x_scale
Y = torch.randn(n_samples, d) * y_scale




cka = linear_cka_efficient(X, Y)
print(f"CKA: {cka}")
_, _, procrustes_result = procrustes_pytorch(X, Y)
print(f"Procrustes: {procrustes_result}")
rsa_result, _, _ = rsa_with_rdms(X, Y)
print(f"RSA: {rsa_result}")
cca_result = cca_similarity(X, Y)
print(f"CCA: {cca_result}")
