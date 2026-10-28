#include "dpd/ode/sde.hpp"

#include <thread>
#include <vector>

namespace dpd::ode {

std::string_view to_string(SdeScheme s) noexcept {
    switch (s) {
        case SdeScheme::EulerMaruyama: return "euler_maruyama";
        case SdeScheme::Milstein: return "milstein";
        case SdeScheme::LinearImplicitEuler: return "linear_implicit_euler";
    }
    return "?";
}

namespace {
std::uint64_t splitmix64(std::uint64_t& x) noexcept {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
inline std::uint64_t rotl(std::uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
}  // namespace

Rng::Rng(std::uint64_t seed, std::uint64_t stream) {
    std::uint64_t x = seed ^ (stream * 0xD1B54A32D192ED03ULL + 0x632BE59BD9B4E019ULL);
    for (auto& s : s_) s = splitmix64(x);
}

std::uint64_t Rng::next() noexcept {
    const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
}

Real Rng::uniform() noexcept { return (static_cast<Real>(next() >> 11) + 0.5) * 0x1.0p-53; }

Real Rng::normal() noexcept {
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }
    const Real u1 = uniform(), u2 = uniform();
    const Real rad = std::sqrt(-2 * std::log(u1));
    const Real ang = 2 * 3.14159265358979323846 * u2;
    spare_ = rad * std::sin(ang);
    has_spare_ = true;
    return rad * std::cos(ang);
}

namespace {
struct Work {
    std::vector<Real> f, g, df, dg, yp, ym, fp, fm, dW;
    explicit Work(Index n) : f(n), g(n), df(n), dg(n), yp(n), ym(n), fp(n), fm(n), dW(n) {}
};

// Central differences of the diagonal derivatives d f_i/d y_i, d g_i/d y_i.
void diag_derivs(const SdeSystem& s, Real t, std::span<const Real> y, Work& w, bool need_f, bool need_g) {
    const Index n = y.size();
    if (s.has_derivatives()) {
        if (need_f) s.drift_diag_jacobian(t, y, w.df);
        if (need_g) s.diffusion_derivative(t, y, w.dg);
        return;
    }
    for (Index i = 0; i < n; ++i) {
        std::copy(y.begin(), y.end(), w.yp.begin());
        std::copy(y.begin(), y.end(), w.ym.begin());
        const Real h = 1e-6 * std::max(std::abs(y[i]), Real{1e-8});
        w.yp[i] += h;
        w.ym[i] -= h;
        const Real dh = w.yp[i] - w.ym[i];
        if (need_f) {
            s.drift(t, w.yp, w.fp);
            s.drift(t, w.ym, w.fm);
            w.df[i] = (w.fp[i] - w.fm[i]) / dh;
        }
        if (need_g) {
            s.diffusion(t, w.yp, w.fp);
            s.diffusion(t, w.ym, w.fm);
            w.dg[i] = (w.fp[i] - w.fm[i]) / dh;
        }
    }
}

bool step(const SdeSystem& s, SdeScheme scheme, Real t, std::span<Real> y, Real h, Rng& rng, Work& w,
          std::span<Real> W) {
    const Index n = y.size();
    s.drift(t, y, w.f);
    s.diffusion(t, y, w.g);
    const Real sq = std::sqrt(h);
    for (Index i = 0; i < n; ++i) w.dW[i] = sq * rng.normal();
    if (!W.empty())
        for (Index i = 0; i < n; ++i) W[i] += w.dW[i];
    switch (scheme) {
        case SdeScheme::EulerMaruyama:
            for (Index i = 0; i < n; ++i) y[i] += w.f[i] * h + w.g[i] * w.dW[i];
            break;
        case SdeScheme::Milstein:
            diag_derivs(s, t, y, w, false, true);
            for (Index i = 0; i < n; ++i)
                y[i] += w.f[i] * h + w.g[i] * w.dW[i] + 0.5 * w.g[i] * w.dg[i] * (w.dW[i] * w.dW[i] - h);
            break;
        case SdeScheme::LinearImplicitEuler:
            diag_derivs(s, t, y, w, true, false);
            for (Index i = 0; i < n; ++i) {
                const Real denom = 1 - h * std::min(w.df[i], Real{0});  // implicit only for dissipative drift
                y[i] += (w.f[i] * h + w.g[i] * w.dW[i]) / denom;
            }
            break;
    }
    for (Index i = 0; i < n; ++i)
        if (!std::isfinite(y[i])) return false;
    return true;
}
}  // namespace

bool simulate_path(const SdeSystem& sys, SdeScheme scheme, Real t0, std::span<Real> y, Real t_end, Real dt, Rng& rng,
                   std::span<Real> W) {
    Work w(sys.size());
    Real t = t0;
    while (t < t_end) {
        const Real h = std::min(dt, t_end - t);
        if (!step(sys, scheme, t, y, h, rng, w, W)) return false;
        t = (t_end - t <= dt) ? t_end : t + h;
    }
    return true;
}

SdeResult simulate(const SdeSystem& sys, Real t0, std::span<const Real> y0, const SdeOptions& opt) {
    SdeResult res;
    const Index n = sys.size();
    res.n = n;
    res.n_paths = opt.n_paths;
    res.t_out = opt.t_out;
    if (n == 0 || y0.size() != n || !(opt.dt > 0) || opt.n_paths == 0 || opt.t_out.empty() ||
        !std::is_sorted(opt.t_out.begin(), opt.t_out.end()) || !(opt.t_out.front() >= t0)) {
        res.message = "invalid input: need y0 of size n, dt > 0, n_paths > 0 and sorted t_out >= t0";
        return res;
    }
    const std::size_t K = opt.t_out.size();
    res.y.assign(opt.n_paths * K * n, Real{0});
    std::atomic<std::size_t> next{0}, failed{0};
    std::atomic<std::uint64_t> steps{0};
    std::atomic<bool> cancelled{false};
    auto worker = [&] {
        Work w(n);
        std::vector<Real> y(n);
        for (std::size_t p; (p = next.fetch_add(1)) < opt.n_paths;) {
            if (opt.cancel && opt.cancel->load(std::memory_order_relaxed)) {
                cancelled = true;
                return;
            }
            Rng rng(opt.seed, p);
            std::copy(y0.begin(), y0.end(), y.begin());
            Real t = t0;
            bool ok = true;
            std::uint64_t local = 0;
            for (std::size_t k = 0; k < K; ++k) {
                const Real te = opt.t_out[k];
                while (ok && t < te) {
                    const Real h = std::min(opt.dt, te - t);
                    ok = step(sys, opt.scheme, t, y, h, rng, w, {});
                    t = (te - t <= opt.dt) ? te : t + h;
                    ++local;
                }
                Real* out = res.y.data() + (p * K + k) * n;
                for (Index i = 0; i < n; ++i) out[i] = ok ? y[i] : std::numeric_limits<Real>::quiet_NaN();
            }
            if (!ok) failed.fetch_add(1);
            steps.fetch_add(local);
        }
    };
    unsigned th = opt.threads ? opt.threads : std::max(1u, std::thread::hardware_concurrency());
    th = static_cast<unsigned>(std::min<std::size_t>(th, opt.n_paths));
    if (th <= 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        for (unsigned k = 0; k < th; ++k) pool.emplace_back(worker);
        for (auto& t : pool) t.join();
    }
    res.n_failed_paths = failed.load();
    res.n_steps = steps.load();
    if (cancelled) {
        res.status = Status::Cancelled;
        res.message = "cancelled by caller";
    } else if (res.n_failed_paths) {
        res.status = Status::NonFiniteState;
        res.message = std::to_string(res.n_failed_paths) + " paths produced non-finite values (stored as NaN)";
    } else {
        res.status = Status::Success;
    }
    return res;
}

}  // namespace dpd::ode
