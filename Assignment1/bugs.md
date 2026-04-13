ugs
1. Cell 42 — stale comment

patience = 100  # stop after 1 check interval (100 steps) with no val improvement
The comment is from when patience was 1. With patience=100 and checks every 100 steps, it now stops after 10,000 steps (100 × 100) of no improvement, not 100. Comment needs updating.

2. Cell 35 — non-reproducible data split

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, shuffle=True, stratify=y_temp)
No random_state — every run gets a different split. The assignment says samples must be randomly chosen, but reproducibility is expected in a submitted notebook. Should use random_state=SEED.

3. Cell 21 — dead commented-out code

# Y = Y.reshape(AL.shape) #
Should be removed.

Doesn't Follow Assignment Spec
4. Cell 13 — batchnorm applied AFTER activation, but cache doesn't include it

A, cache = linear_activation_forward(A_prev, W, b, activation="relu")
if use_batchnorm:
    A = apply_batchnorm(A)
caches.append(cache)   # ← cache stores pre-batchnorm A
The next layer receives post-batchnorm A as its input, but cache stores the pre-batchnorm version. During backprop, dW for the next layer is computed using the wrong A_prev. This means gradients flow through batchnorm but batchnorm itself is not backpropagated through.

The assignment (Section 5) says: "There is no need to update the parameters of the batchnorm" — it doesn't say ignore its effect on gradients. That said, this is a common simplification in educational assignments and won't cause training to completely fail.

5. Cell 15 / Cell 22 — compute_cost and update_parameters signatures differ from spec
The assignment specifies:

compute_cost(AL, Y) — no L2 params
update_parameters(parameters, grads, learning_rate) — no lambda/m
You added lambda_reg and m for Section 6 (L2). This is fine per rule (d): "You are allowed to implement auxiliary functions in addition to those defined." But you should mention and explain it in the report.

Minor Issues
6. Cell 18 — unnecessary isinstance check in relu_backward

Z = activation_cache["Z"] if isinstance(activation_cache, dict) else activation_cache
The cache is always a dict — the check is defensive code that can never help. Can be simplified to Z = activation_cache["Z"].

Summary Table
Cell	Severity	Issue
42	Medium	Stale comment on patience
35	Medium	No random_state → non-reproducible split
21	Low	Dead commented-out code
13	Medium	Batchnorm not captured in cache → approximate gradients
15/22	Low	Signature deviates from spec — must mention in report
18	Low	Unnecessary isinstance check
Want me to fix all of these?