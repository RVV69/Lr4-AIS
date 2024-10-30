import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve, train_test_split

np.random.seed(0)
X = np.sort(6 * np.random.rand(100, 1) - 3, axis=0)
y = 0.5 * X ** 2 + X + 2 + np.random.randn(100, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def plot_learning_curve(model, X_train, y_train, title):
    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5,
                                                           scoring='neg_mean_squared_error',
                                                           train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, label="Training error")
    plt.plot(train_sizes, val_scores_mean, label="Validation error")
    plt.title(title)
    plt.xlabel("Training set size")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid()
    plt.show()


linear_model = LinearRegression()
plot_learning_curve(linear_model, X_train, y_train, "Learning Curve (Linear Regression)")

poly10_model = make_pipeline(PolynomialFeatures(10), LinearRegression())
plot_learning_curve(poly10_model, X_train, y_train, "Learning Curve (Polynomial Regression, Degree 10)")

poly2_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
plot_learning_curve(poly2_model, X_train, y_train, "Learning Curve (Polynomial Regression, Degree 2)")

