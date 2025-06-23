Logistic Regression from Scratch (with Linear & Polynomial Decision Boundaries)

# 🧠 Logistic Regression from Scratch (with Linear and Polynomial Boundaries)

This project demonstrates how to implement **Logistic Regression from scratch using NumPy** for binary classification, including both **linear** and **polynomial decision boundaries**. It uses synthetic 2D data generated with `sklearn.datasets.make_classification`.

---

## 📌 Project Highlights

- Logistic regression (sigmoid-based binary classifier)
- Gradient descent optimization
- Polynomial feature expansion for non-linear separation
- Manual cost function and gradient calculations
- Visualization of decision boundaries (linear & non-linear)

---

## 📊 Dataset

- Created using `make_classification` from scikit-learn
- 2 informative features
- 2 classes with some label noise (10%)
- 200 total samples

```python
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,
    flip_y=0.1,
    random_state=42
)
