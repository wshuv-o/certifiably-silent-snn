"""Fixed-point golden model of a time-stepped recurrent ALIF network with multi-tap delays, and of the
per-core silence certificate as implemented in hardware (harness/hw/core.py).

All state is integer, Q(F) with F fractional bits. Multiplications by the constants beta, rho are
floor((c * x) >> F), so every operation is monotone in its operands; the certificate, which applies the same
operations to upper bounds, is therefore sound in fixed point by construction (and checked empirically).

Hardware certificate (no fixed-point refinement, all presynaptic neurons may fire): rooted after step t, for
k = 1..K, u_k = floor(beta*u_{k-1}) + I(t+k) + acc(t+k) + B_k, where acc(t+k) is the delay accumulator slot for
step t+k (holding exactly the contributions of spikes already emitted) and B_k = sum_{d<k} R^(d) with R^(d) the
quantised sum of positive weights into the neuron through tap d. The threshold at t+k is the exact adaptive
threshold until a first spike: theta + a_k with a_1 = floor(rho*a') + gamma*s', a_k = floor(rho*a_{k-1}).
"""
import numpy as np

F = 12                                   # fractional bits
ONE = 1 << F


def q(x):
    return np.round(np.asarray(x, dtype=np.float64) * ONE).astype(np.int64)


def mulc(c_fx, x):
    """floor(c * x) in Q(F); c_fx >= 0."""
    return (c_fx * x) >> F


class Net:
    def __init__(self, W, D, I, beta, theta, rho, gamma, C):
        self.W = [q(w) for w in W]                         # (H, H) post x pre, per tap
        self.D = list(D); self.I = q(I)                    # (N, T, H)
        self.beta, self.theta, self.rho, self.gamma = q(beta), q(theta), q(rho), q(gamma)
        self.H = self.W[0].shape[0]; self.C = C; self.P = self.H // C
        self.Rd = [np.maximum(w, 0).sum(1) for w in self.W]  # (H,) per tap, exact in Q(F)

    def bounded(self, k):
        return sum(r for r, d in zip(self.Rd, self.D) if d < k)

    def run(self, b, K=4, adapt=True):
        """Returns spikes S (T,H), certificates cert (T,P) [core p certified silent at t+1..t+K], and the
        per-step inputs the hardware core sees: v_post (T,H), a (T,H), acc slots (T, K, H)."""
        T = self.I.shape[1]; H, C, P = self.H, self.C, self.P; DMAX = max(self.D)
        v = np.zeros(H, np.int64); a = np.zeros(H, np.int64); s = np.zeros(H, np.int64)
        acc = np.zeros((T + DMAX + K + 1, H), np.int64)    # delay accumulator, slot = arrival step
        S = np.zeros((T, H), np.int64); V = np.zeros((T, H), np.int64); A = np.zeros((T, H), np.int64)
        cert = np.zeros((T, P), bool); viol = 0; slots = np.zeros((T, K, H), np.int64)
        for t in range(1, T):
            a = mulc(self.rho, a) + self.gamma * s
            u = mulc(self.beta, v) + self.I[b, t] + acc[t]
            thr = self.theta + a
            s = (u >= thr).astype(np.int64)
            v = u - s * thr
            S[t], V[t], A[t] = s, v, a
            for w, d in zip(self.W, self.D):                   # synaptic events of step-t spikes
                if s.any():
                    acc[t + d] += w @ s
            # certificate rooted after step t (uses only spikes up to t: acc slots t+1..t+K are exact parts)
            if t + K < T:
                uk = v.copy(); ak = a.copy(); fire = np.zeros(H, bool)
                for k in range(1, K + 1):
                    slots[t, k - 1] = acc[t + k]
                    uk = mulc(self.beta, uk) + self.I[b, t + k] + acc[t + k] + self.bounded(k)
                    ak = (mulc(self.rho, ak) + self.gamma * s) if k == 1 else mulc(self.rho, ak)
                    fire |= uk >= (self.theta + (ak if adapt else 0))
                cert[t] = ~fire.reshape(P, C).any(1)
        # soundness: a certified core must not spike at t+1..t+K
        for t in range(1, T - K):
            for p in np.nonzero(cert[t])[0]:
                viol += int(S[t + 1:t + K + 1, p * C:(p + 1) * C].sum())
        return S, cert, viol, V, A, slots
