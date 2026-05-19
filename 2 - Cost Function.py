# -----------------------------------------------------------------------
# LAB: Cost Function for Linear Regression (One Variable)
# -----------------------------------------------------------------------
# The goal here is to build and understand J(w,b) — the cost function.
# It tells us how wrong our model is. Lower cost = better fit.
# -----------------------------------------------------------------------

# imports
# numpy      — for math and arrays
# matplotlib — for plotting
# lab_utils_uni — course-provided helpers for the interactive plots
# %matplotlib widget — makes plots interactive inside Jupyter (sliders, clicks)
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
import matplotlib.pyplot as plt
import numpy as np
%matplotlib widget
plt.style.use('./deeplearning.mplstyle')  # just makes the plots look nicer


# -----------------------------------------------------------------------
# the data — two houses, that's it
# -----------------------------------------------------------------------
# house 1: 1000 sqft sold for $300k
# house 2: 2000 sqft sold for $500k
# everything's in the thousands to keep numbers clean
x_train = np.array([1.0, 2.0])      # size (1000 sqft)
y_train = np.array([300.0, 500.0])  # price ($1000s)


# -----------------------------------------------------------------------
# the cost function J(w, b)
# -----------------------------------------------------------------------
# our model is just a straight line: f(x) = w*x + b
#   w = slope (how much the price goes up per unit of size)
#   b = intercept (the starting price before size even matters)
#
# the cost function measures how far off our predictions are, on average:
#
#   J(w,b) = (1 / 2m) * sum of (f(x[i]) - y[i])^2  for all i
#
# why square the errors?
#   so negatives and positives don't cancel, and big mistakes get punished harder
#
# why divide by 2m?
#   dividing by m gives the average error across all examples
#   the extra /2 is just a calculus shortcut — it cancels out a 2 that
#   pops up when you differentiate later in gradient descent
#
# the job: find the w and b that make J as small as possible

def compute_cost(x, y, w, b):
    """
    Computes the cost J(w,b) for linear regression.

    Args:
      x (ndarray, shape (m,)): input features, one per training example
      y (ndarray, shape (m,)): true target values
      w (scalar)             : slope of the model line
      b (scalar)             : intercept of the model line

    Returns:
      total_cost (float): how wrong the model currently is (lower is better)
    """

    # how many training examples do we have?
    m = x.shape[0]

    cost_sum = 0  # we'll accumulate the squared errors here

    for i in range(m):
        # step 1: make a prediction for this example
        f_wb = w * x[i] + b

        # step 2: how wrong were we? square it so sign doesn't matter
        cost = (f_wb - y[i]) ** 2

        # step 3: add it to the running total
        cost_sum = cost_sum + cost

    # average it out and apply the /2 trick
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


# -----------------------------------------------------------------------
# plot 1 — cost vs. w with a slider (b is locked at 100)
# -----------------------------------------------------------------------
# b=100 was already found to be a good value in the previous lab,
# so here we just move w around and see what happens to the cost.
#
# things to notice:
#   - cost bottoms out at w=200, which gives a perfect fit for our 2 points
#   - the curve is U-shaped, so there's exactly one minimum (no tricks)
#   - moving w too far in either direction sends the cost sky high fast
#
# drag the slider yourself and watch the line try to fit the data

plt_intuition(x_train, y_train)


# -----------------------------------------------------------------------
# bigger dataset — 6 points that don't sit on a line
# -----------------------------------------------------------------------
# this is more realistic. real data is messy.
# with scatter like this, we can't get cost all the way to 0 —
# we're just looking for the best possible fit

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250,  300,  480,  430,  630,  730])


# -----------------------------------------------------------------------
# plot 2 — contour map of J(w, b) — click around to explore
# -----------------------------------------------------------------------
# now both w and b are free, so the cost surface is 2D
#
# left plot: the data + model line. the dashed vertical lines show
#   how much each point is contributing to the total cost right now
#
# right plot: a contour map of J(w,b) viewed from above (like a topographic map)
#   the innermost rings = lowest cost
#   click anywhere on it to pick a (w, b) and see how the line changes
#
# best spot here is around w ≈ 209, b ≈ 2.4
# even there, cost won't be zero because the data just doesn't line up perfectly

plt.close('all')  # close old figures so things don't get cluttered
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
# updater is what makes clicking on the contour actually do something


# -----------------------------------------------------------------------
# plot 3 — the soup bowl (why convexity is a big deal)
# -----------------------------------------------------------------------
# this is J(w,b) shown as a 3D surface with w and b on equal scales
#
# because we square the errors, the surface is always convex — smooth,
# bowl-shaped, with one single lowest point
#
# this is important because it means gradient descent will always
# find the real minimum. no getting stuck in a false bottom.
# it's one of the main reasons squared error is used for linear regression

soup_bowl()
