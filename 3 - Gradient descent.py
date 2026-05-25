"""
Gradient Descent for Linear Regression
=======================================
This script walks through implementing gradient descent from scratch
to fit a simple linear regression model. No frameworks, just numpy math.

We're using a tiny housing dataset — 2 data points — to keep things clear.
The goal is to find the best values for w (slope) and b (intercept) so that
our model f(x) = wx + b predicts house prices as accurately as possible.
"""

import math
import numpy as np

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
# Two houses: size in 1000 sqft, price in $1000s
# House 1: 1000 sqft → $300k
# House 2: 2000 sqft → $500k
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


# ─────────────────────────────────────────────
# COST FUNCTION  →  J(w, b)
# ─────────────────────────────────────────────
# This measures how wrong our current model is.
# We square the errors so negatives don't cancel positives out,
# and divide by 2m to keep the math clean when we differentiate later.
#
# J(w,b) = (1 / 2m) * Σ (f(x_i) - y_i)²
def compute_cost(x, y, w, b):
    """
    Calculates mean squared error cost for given w and b.

    Args:
        x: input features (array)
        y: actual target values (array)
        w: weight / slope
        b: bias / intercept

    Returns:
        total_cost: a single number — lower is better
    """
    m = x.shape[0]
    cost = 0

    for i in range(m):
        prediction = w * x[i] + b
        cost += (prediction - y[i]) ** 2

    total_cost = cost / (2 * m)
    return total_cost


# ─────────────────────────────────────────────
# GRADIENT FUNCTION  →  dJ/dw and dJ/db
# ─────────────────────────────────────────────
# This tells us the slope of the cost curve at our current w and b.
# Gradient descent needs this to know which direction to step.
#
# dJ/dw = (1/m) * Σ (f(x_i) - y_i) * x_i
# dJ/db = (1/m) * Σ (f(x_i) - y_i)
def compute_gradient(x, y, w, b):
    """
    Computes partial derivatives of the cost w.r.t. w and b.

    Args:
        x: input features (array)
        y: actual target values (array)
        w: current weight
        b: current bias

    Returns:
        dj_dw: gradient with respect to w
        dj_db: gradient with respect to b
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        prediction = w * x[i] + b
        error = prediction - y[i]

        # Each example contributes a bit to the overall gradient
        dj_dw += error * x[i]
        dj_db += error

    # Average over all examples
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


# ─────────────────────────────────────────────
# GRADIENT DESCENT
# ─────────────────────────────────────────────
# Repeatedly nudge w and b in the direction that reduces cost.
# "alpha" is the learning rate — how big each nudge is.
# Too small → very slow. Too large → overshoots and diverges.
#
# Update rule (applied simultaneously):
#   w = w - alpha * dJ/dw
#   b = b - alpha * dJ/db
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Runs gradient descent for num_iters steps.

    Args:
        x:         input features
        y:         target values
        w_in:      starting value for w
        b_in:      starting value for b
        alpha:     learning rate
        num_iters: how many steps to take

    Returns:
        w:         final optimized weight
        b:         final optimized bias
        J_history: cost at each iteration (useful for plotting / debugging)
        p_history: [w, b] at each iteration
    """
    J_history = []
    p_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        # Figure out which direction is "downhill"
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Take a step in that direction
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Record what happened this iteration
        if i < 100_000:  # guard against runaway memory usage on long runs
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])

        # Print a progress update every 10% of the way through
        if i % math.ceil(num_iters / 10) == 0:
            cost_now = J_history[-1]
            print(
                f"  Iteration {i:5d} | "
                f"Cost: {cost_now:10.4f} | "
                f"dj_dw: {dj_dw:+.4f}  dj_db: {dj_db:+.4f} | "
                f"w: {w:.4f}  b: {b:.4f}"
            )

    return w, b, J_history, p_history


# ─────────────────────────────────────────────
# MAIN — put it all together
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Gradient Descent for Linear Regression")
    print("=" * 65)

    # ── Initial cost check ──────────────────────────────────────────
    # Before training, w=0 and b=0, so every prediction is just 0.
    # The cost here should be very high — we haven't learned anything yet.
    w_init, b_init = 0, 0
    initial_cost = compute_cost(x_train, y_train, w_init, b_init)
    print(f"\nInitial cost (w=0, b=0): {initial_cost:.2f}")
    print("  → This is the cost before any training. We expect it to be high.\n")

    # ── Good learning rate run ───────────────────────────────────────
    print("─" * 65)
    print("  RUN 1: Standard learning rate (alpha = 0.01, 10,000 iterations)")
    print("─" * 65)

    iterations = 10_000
    alpha = 1.0e-2

    w_final, b_final, J_hist, p_hist = gradient_descent(
        x_train, y_train, w_init, b_init, alpha, iterations
    )

    print(f"\n✓ Training complete!")
    print(f"  Optimal w = {w_final:.4f}")
    print(f"  Optimal b = {b_final:.4f}")
    print(f"  Final cost = {J_hist[-1]:.6f}")

    # ── Predictions with learned parameters ─────────────────────────
    # These should be close to the training data ($300k and $500k),
    # and the 1200 sqft prediction should fall in between.
    print("\n  Predictions using learned model:")
    print(f"    1000 sqft house → ${w_final * 1.0 + b_final:.1f}k")
    print(f"    1200 sqft house → ${w_final * 1.2 + b_final:.1f}k  (unseen — should be ~$340k)")
    print(f"    2000 sqft house → ${w_final * 2.0 + b_final:.1f}k")

    # ── How the cost changed over training ──────────────────────────
    print("\n  Cost history snapshot (every 1000 iterations):")
    for step in range(0, len(J_hist), 1000):
        print(f"    Iteration {step:5d}: cost = {J_hist[step]:.4f}")

    # ── Too-large learning rate run ──────────────────────────────────
    print("\n" + "─" * 65)
    print("  RUN 2: Learning rate WAY too large (alpha = 0.8, 10 iterations)")
    print("  Watch what happens — cost should grow instead of shrink!")
    print("─" * 65)

    bad_alpha = 8.0e-1
    bad_iters = 10

    w_bad, b_bad, J_hist_bad, _ = gradient_descent(
        x_train, y_train, w_init, b_init, bad_alpha, bad_iters
    )

    print(f"\n  Final w = {w_bad:.4f},  b = {b_bad:.4f}")
    print(f"  Final cost = {J_hist_bad[-1]:.2f}")
    print("  → Notice the cost exploded. That's divergence from too large an alpha.\n")

    # ── Summary ─────────────────────────────────────────────────────
    print("=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"  Best w found:   {w_final:.4f}")
    print(f"  Best b found:   {b_final:.4f}")
    print(f"  Final cost:     {J_hist[-1]:.8f}  (basically zero — perfect fit)")
    print(f"\n  The model f(x) = {w_final:.2f}x + {b_final:.2f} fits the training data well.")
    print("  Key takeaway: learning rate matters a lot. Too big → diverge. Too small → slow.")
    print()


if __name__ == "__main__":
    main()
