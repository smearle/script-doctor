#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <cmath>

// RC4-based seedable PRNG matching PuzzleScript's rng.js exactly.
struct RC4 {
    uint8_t s[256];
    int i = 0;
    int j = 0;

    RC4() {
        for (int k = 0; k < 256; ++k) s[k] = static_cast<uint8_t>(k);
    }

    void mix(const uint8_t* key, int keylen) {
        int j_mix = 0;
        for (int k = 0; k < 256; ++k) {
            j_mix = (j_mix + s[k] + key[k % keylen]) & 0xff;
            uint8_t tmp = s[k];
            s[k] = s[j_mix];
            s[j_mix] = tmp;
        }
        // Advance past initial bytes (same as JS implementation)
        for (int k = 0; k < 256; ++k) next();
    }

    uint8_t next() {
        i = (i + 1) & 0xff;
        j = (j + s[i]) & 0xff;
        uint8_t tmp = s[i];
        s[i] = s[j];
        s[j] = tmp;
        return s[(s[i] + s[j]) & 0xff];
    }
};

struct RNG {
    RC4 rc4;
    bool has_normal = false;
    double cached_normal = 0.0;

    void seed(const std::string& seed_str) {
        has_normal = false;
        cached_normal = 0.0;
        rc4 = RC4();
        if (seed_str.empty()) {
            // No seed: use zeros
            uint8_t zero[1] = {0};
            rc4.mix(zero, 1);
        } else {
            std::vector<uint8_t> key(seed_str.begin(), seed_str.end());
            rc4.mix(key.data(), static_cast<int>(key.size()));
        }
    }

    // Returns a double in [0, 1) matching JS RNG.uniform()
    // Uses 7 bytes to produce a 56-bit mantissa (same as JS)
    double uniform() {
        double x = 0.0;
        double denom = 1.0;
        for (int k = 0; k < 7; ++k) {
            x = x * 256.0 + rc4.next();
            denom *= 256.0;
        }
        return x / denom;
    }

    int random_int(int n) {
        return static_cast<int>(std::floor(uniform() * n));
    }
};
