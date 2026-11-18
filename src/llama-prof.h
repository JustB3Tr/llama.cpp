#pragma once
// LLAMA_HOST_PROF=1: wall time per named host-side region, summed and printed at exit with the median (lc- debugging aid)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct llama_prof_reg {
    struct entry { double sum = 0.0; long n = 0; std::vector<float> v; };
    std::map<std::string, entry> acc;
    std::mutex mu;
    ~llama_prof_reg() {
        for (auto & kv : acc) {
            auto & v = kv.second.v;
            float med = 0.0f;
            if (!v.empty()) {
                std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
                med = v[v.size()/2];
            }
            fprintf(stderr, "lc-prof %-36s %10.1f ms %8ld calls %9.1f us/call  median %9.1f us\n", kv.first.c_str(), kv.second.sum/1e3,
                    kv.second.n, kv.second.sum/kv.second.n, med);
        }
    }
};

inline llama_prof_reg & llama_prof() { static llama_prof_reg r; return r; }
inline bool llama_prof_on() { static const bool on = getenv("LLAMA_HOST_PROF") != nullptr; return on; }

inline void llama_prof_add(const char * name, double us) {
    auto & r = llama_prof();
    std::lock_guard<std::mutex> lock(r.mu);
    auto & e = r.acc[name];
    e.sum += us; e.n++;
    if (e.v.size() < 1000000) { e.v.push_back((float) us); }
}

// laps: a chain of host timestamps from one event to another (lap.<name> = time since the previous lap point); begun
// by llama_prof_lap_begin, ended by a lap with end = true. Used for verify end -> first draft launch.
struct llama_prof_lap_state { std::chrono::steady_clock::time_point t; bool on = false; };
inline llama_prof_lap_state & llama_prof_laps() { static llama_prof_lap_state s; return s; }
inline void llama_prof_lap_begin() {
    if (!llama_prof_on()) { return; }
    auto & s = llama_prof_laps();
    s.t = std::chrono::steady_clock::now(); s.on = true;
}
inline void llama_prof_lap(const char * name, bool end = false) {
    if (!llama_prof_on()) { return; }
    auto & s = llama_prof_laps();
    if (!s.on) { return; }
    const auto now = std::chrono::steady_clock::now();
    llama_prof_add(name, std::chrono::duration<double, std::micro>(now - s.t).count());
    s.t = now;
    if (end) { s.on = false; }
}

struct llama_prof_scope {
    const char * name;
    bool on;
    std::chrono::steady_clock::time_point t0;
    explicit llama_prof_scope(const char * n) : name(n), on(llama_prof_on()) {
        if (on) { t0 = std::chrono::steady_clock::now(); }
    }
    ~llama_prof_scope() {
        if (!on) { return; }
        llama_prof_add(name, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
};

#define LC_PROF_CAT2(a, b) a##b
#define LC_PROF_CAT(a, b) LC_PROF_CAT2(a, b)
#define LC_PROF(name) llama_prof_scope LC_PROF_CAT(lc_prof_, __LINE__)(name)
