import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold

# Xây dựng model cho Linear Regession
##Clone của P Hiếu
class LinearRegression:
    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.beta = None
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.selector = None
        self.scaler = StandardScaler()
        self.weakf = [ 'Resource']  # tên các cột yếu để giữ lại 'Interface', 'File',

    def fit(self, X, y):
        # Tách weak features
        X_weak = X[self.weakf].to_numpy()
        X_main = X.drop(columns=self.weakf).to_numpy()

        # Nhân đa thức
        X_poly = self.poly.fit_transform(X_main)

        #Lấy k đặc trưng sau khi nhân giá trị
        self.selector = SelectKBest(score_func=f_regression, k=8)
        X_poly_selected = self.selector.fit_transform(X_poly, y.ravel())

        #Ghép k đặc trưng mới với các đặc trưng yếu
        X_final = np.concatenate((X_poly_selected, X_weak), axis=1)
        # X_scaled = self.scaler.fit_transform(X_final)

        # Ridge regression closed-form
        X_b = np.c_[np.ones((X_final.shape[0], 1)), X_final]
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        m, n = X_b.shape
        I = np.eye(n)
        I[0, 0] = 0
        self.beta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y

    def transform_X(self, X):
        X_weak = X[self.weakf].to_numpy()
        X_main = X.drop(columns=self.weakf).to_numpy()
        X_poly = self.poly.transform(X_main)
        X_poly_selected = self.selector.transform(X_poly)

        X_final = np.concatenate((X_poly_selected, X_weak), axis=1)
        # return self.scaler.transform(X_final)
        return X_final

    def predict(self, X):
        X_scaled = self.transform_X(X)
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return X_b @ self.beta

    def score(self, X, y):
        y_pred = self.predict(X)
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        return r2_score(y, y_pred)

def ridge_regression(X_train, y_train):
    """Tìm alpha tốt nhất cho Ridge Regression với cross-validation."""
    alphas = np.logspace(-1, 4, 100)
    best_alpha = None
    best_r2 = -np.inf

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for alpha in alphas:
        r2_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
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

    # Train final model với alpha tốt nhất
    final_model = LinearRegression(alpha=best_alpha)
    final_model.fit(X_train, y_train)

    print("Beta Coefficients:")
    print(final_model.beta)

    return final_model