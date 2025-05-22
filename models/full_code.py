import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold

# Xây dựng model cho Linear Regession
class LinearRegression:
    def __init__(self, alpha):
        self.alpha = alpha  # Hệ số điều chuẩn λ
        self.beta = None
        self.poly_transformer = None # Lưu các đặc trưng phi tuyến từ các đặc trưng gốc.
        self.selector = None  # Lưu bộ đặc trưng tối ưu được chọn
        self.scaler = StandardScaler()

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]
    def print_beta(self):
        return self.beta

    def fit(self, X, y):
        # Chuẩn hóa X
        X_scaled = self.scaler.fit_transform(X)
        X_b = self._add_bias(X_scaled)

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        m, n = X_b.shape

        # Tính ω* = (X^T X + αI)^(-1) X^T y
        I = np.eye(n)
        I[0, 0] = 0  # Không regularize bias
        self.beta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_b = self._add_bias(X_scaled)
        return X_b @ self.beta

    def score(self, X, y_true):
        y_pred = self.predict(X)
        y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
        return r2_score(y_true, y_pred)

# Xây dựng thuật toán ridge regression để huấn luyện mô hình
def ridge_regression(X_train, y_train, X_test, y_test, use_feature_selection=False):
    # Bước 1: Biến đổi đa thức
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    selector = None
    if use_feature_selection:
        # Chọn ra 9 đặc trưng có mức độ ảnh hưởng nhất đến nhãn thông qua hàm đánh giá điểm f_regression
        selector = SelectKBest(score_func=f_regression, k=9)
        X_train_poly = selector.fit_transform(X_train_poly, y_train.ravel())
        X_test_poly = selector.transform(X_test_poly)

    # Bước 2: Cross-validation với mô hình LinearRegression tự xây
    # Tìm hệ số alpha tối ưu nhất
    alphas = np.logspace(-3, 3, 20)
    best_alpha = None
    best_r2 = -np.inf

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for alpha in alphas:
        r2_scores = []
        for train_idx, val_idx in kf.split(X_train_poly):
            X_tr, X_val = X_train_poly[train_idx], X_train_poly[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = LinearRegression(alpha=alpha)
            model.fit(X_tr, y_tr)
            r2 = model.score(X_val, y_val)
            r2_scores.append(r2)

        avg_r2 = np.mean(r2_scores)
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_alpha = alpha

    print(f"\n🔍 Best alpha: {best_alpha:.4f}")

    # Bước 3: Train lại mô hình tốt nhất & Lưu lại các đặc tính được chọn để tiện cho tính toán
    final_model = LinearRegression(alpha=best_alpha)
    final_model.poly_transformer = poly
    final_model.selector = selector
    final_model.fit(X_train_poly, y_train)
    print("Giá trị hệ số beta:")
    print(final_model.print_beta())

    return final_model, selector, poly
