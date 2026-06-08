"""SINDy/SINDYc model utilities for FANUC position-servo dynamics."""

import numpy as np


class SINDYcModel:
    """
    Controlled SINDy model:
        x_dot = Theta(x, u) Xi

    x = [q1..q6, dq1..dq6]
    u = [q1_cmd..q6_cmd]
    """

    def __init__(self):
        self.Xi = None
        self.x_mean = None
        self.x_std = None
        self.u_mean = None
        self.u_std = None
        self.xdot_mean = None
        self.xdot_std = None
        self.is_fit = False
        self.use_fallback = False
        self.K_servo = np.array([8.0, 8.0, 8.0, 10.0, 10.0, 12.0])

    def use_fallback_model(self):
        """Use a simple second-order servo model when no trained SINDy model exists."""
        self.use_fallback = True
        self.is_fit = False

    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.Xi = data["Xi"]
        self.x_mean = data["x_mean"]
        self.x_std = data["x_std"]
        self.u_mean = data["u_mean"]
        self.u_std = data["u_std"]
        self.xdot_mean = data["xdot_mean"]
        self.xdot_std = data["xdot_std"]
        self.is_fit = True
        self.use_fallback = False

    def save(self, filename):
        if not self.is_fit:
            raise RuntimeError("Cannot save: SINDy model has not been fitted.")
        np.savez(
            filename,
            Xi=self.Xi,
            x_mean=self.x_mean,
            x_std=self.x_std,
            u_mean=self.u_mean,
            u_std=self.u_std,
            xdot_mean=self.xdot_mean,
            xdot_std=self.xdot_std,
        )

    @staticmethod
    def _standardize_fit(A):
        mean = A.mean(axis=0, keepdims=True)
        std = A.std(axis=0, keepdims=True) + 1e-9
        return (A - mean) / std, mean, std

    @staticmethod
    def build_library(X, U):
        N, nx = X.shape
        _, nu = U.shape
        features = [np.ones((N, 1))]
        for i in range(nx):
            features.append(X[:, i:i + 1])
        for j in range(nu):
            features.append(U[:, j:j + 1])
        for i in range(nx):
            for j in range(i, nx):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        for i in range(nu):
            for j in range(i, nu):
                features.append((U[:, i] * U[:, j]).reshape(-1, 1))
        for i in range(nx):
            for j in range(nu):
                features.append((X[:, i] * U[:, j]).reshape(-1, 1))
        for i in range(min(6, nx)):
            features.append(np.sin(X[:, i:i + 1]))
            features.append(np.cos(X[:, i:i + 1]))
        return np.hstack(features)

    def fit(self, X, U, X_dot, ridge=1e-6, threshold=1e-4, n_refine=5):
        """
        Fit sparse coefficients using sequential thresholded least squares.
        This avoids requiring scikit-learn at runtime.
        """
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        X_dot = np.asarray(X_dot, dtype=float)

        Xs, self.x_mean, self.x_std = self._standardize_fit(X)
        Us, self.u_mean, self.u_std = self._standardize_fit(U)
        Xds, self.xdot_mean, self.xdot_std = self._standardize_fit(X_dot)

        Theta = self.build_library(Xs, Us)
        n_features = Theta.shape[1]
        nx = X.shape[1]
        Xi = np.zeros((n_features, nx))

        A = Theta.T @ Theta + ridge * np.eye(n_features)
        B = Theta.T @ Xds
        Xi = np.linalg.solve(A, B)

        for _ in range(n_refine):
            small = np.abs(Xi) < threshold
            Xi[small] = 0.0
            for state_i in range(nx):
                big_idx = np.where(np.abs(Xi[:, state_i]) > 0.0)[0]
                if len(big_idx) == 0:
                    continue
                Theta_big = Theta[:, big_idx]
                A_big = Theta_big.T @ Theta_big + ridge * np.eye(len(big_idx))
                B_big = Theta_big.T @ Xds[:, state_i]
                Xi[big_idx, state_i] = np.linalg.solve(A_big, B_big)

        self.Xi = Xi
        self.is_fit = True
        self.use_fallback = False
        return self

    def predict_xdot(self, x, u):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        u = np.atleast_2d(np.asarray(u, dtype=float))

        if self.use_fallback:
            q = x[:, :6]
            dq = x[:, 6:]
            q_cmd = u
            q_dot = dq
            dq_dot = self.K_servo * (q_cmd - q) - 1.5 * dq
            return np.hstack([q_dot, dq_dot])

        if not self.is_fit:
            raise RuntimeError("No SINDy model loaded or fitted.")

        Xs = (x - self.x_mean) / self.x_std
        Us = (u - self.u_mean) / self.u_std
        Theta = self.build_library(Xs, Us)
        xdot_scaled = Theta @ self.Xi
        return xdot_scaled * self.xdot_std + self.xdot_mean

    def step(self, x, u, dt):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        def f(x_local):
            return self.predict_xdot(x_local.reshape(1, -1), u.reshape(1, -1))[0]

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
