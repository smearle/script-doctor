#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

// Dynamic-length bitvector matching PuzzleScript's BitVec.
// Backed by int32_t words to match JS Int32Array semantics exactly.
struct BitVec {
    std::vector<int32_t> data;

    BitVec() = default;
    explicit BitVec(int stride) : data(stride, 0) {}
    BitVec(const int32_t* src, int stride) : data(src, src + stride) {}

    int size() const { return static_cast<int>(data.size()); }

    void iand(const BitVec& other) {
        for (int i = 0; i < size(); ++i)
            data[i] &= other.data[i];
    }

    void ior(const BitVec& other) {
        for (int i = 0; i < size(); ++i)
            data[i] |= other.data[i];
    }

    void iclear(const BitVec& other) {
        for (int i = 0; i < size(); ++i)
            data[i] &= ~other.data[i];
    }

    void inot() {
        for (int i = 0; i < size(); ++i)
            data[i] = ~data[i];
    }

    void ibitset(int ind) {
        int outer = ind >> 5;
        int inner = ind & 0x1f;
        data[outer] |= (1 << inner);
    }

    void ibitclear(int ind) {
        int outer = ind >> 5;
        int inner = ind & 0x1f;
        data[outer] &= ~(1 << inner);
    }

    bool get(int ind) const {
        int outer = ind >> 5;
        int inner = ind & 0x1f;
        return (data[outer] & (1 << inner)) != 0;
    }

    // Extract 5-bit field at bit position `shift`, masked by `mask`.
    // Matches JS: getshiftor(mask, shift)
    int getshiftor(int mask, int shift) const {
        int inner = shift & 0x1f;
        int outer = shift >> 5;
        int ret = static_cast<uint32_t>(data[outer]) >> inner;
        if (inner > 27) {
            ret |= data[outer + 1] << (32 - inner);
        }
        return ret & mask;
    }

    // OR a masked value into the bitvec at bit position `shift`.
    void ishiftor(int mask, int shift) {
        int inner = shift & 0x1f;
        int outer = shift >> 5;
        data[outer] |= (mask << inner);
        if (inner > 27) {
            data[outer + 1] |= (mask >> (32 - inner));
        }
    }

    // Clear a masked value from the bitvec at bit position `shift`.
    void ishiftclear(int mask, int shift) {
        int inner = shift & 0x1f;
        int outer = shift >> 5;
        data[outer] &= ~(mask << inner);
        if (inner > 27) {
            data[outer + 1] &= ~(mask >> (32 - inner));
        }
    }

    bool iszero() const {
        for (int i = 0; i < size(); ++i)
            if (data[i] != 0) return false;
        return true;
    }

    bool equals(const BitVec& other) const {
        if (size() != other.size()) return false;
        for (int i = 0; i < size(); ++i)
            if (data[i] != other.data[i]) return false;
        return true;
    }

    // Check if all bits set in `this` are also set in `arr`.
    bool bitsSetInArray(const int32_t* arr) const {
        for (int i = 0; i < size(); ++i)
            if ((data[i] & arr[i]) != data[i]) return false;
        return true;
    }

    bool bitsSetInArray(const BitVec& other) const {
        return bitsSetInArray(other.data.data());
    }

    // Check if no bits set in `this` are set in `arr`.
    bool bitsClearInArray(const int32_t* arr) const {
        for (int i = 0; i < size(); ++i)
            if (data[i] & arr[i]) return false;
        return true;
    }

    bool bitsClearInArray(const BitVec& other) const {
        return bitsClearInArray(other.data.data());
    }

    bool anyBitsInCommon(const BitVec& other) const {
        for (int i = 0; i < size(); ++i)
            if (data[i] & other.data[i]) return true;
        return false;
    }

    void setZero() {
        std::memset(data.data(), 0, data.size() * sizeof(int32_t));
    }

    BitVec clone() const {
        BitVec result;
        result.data = data;
        return result;
    }

    void cloneInto(BitVec& target) const {
        for (int i = 0; i < size(); ++i)
            target.data[i] = data[i];
    }
};
