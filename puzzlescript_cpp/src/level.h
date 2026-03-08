#pragma once
#include "bitvec.h"
#include <vector>
#include <cstdint>
#include <string>

struct Level {
    int lineNumber = 0;
    int width = 0;
    int height = 0;
    int n_tiles = 0;
    int layerCount = 0;
    int STRIDE_OBJ = 1;
    int STRIDE_MOV = 1;

    std::vector<int32_t> objects;     // flat: n_tiles * STRIDE_OBJ
    std::vector<int32_t> movements;   // flat: n_tiles * STRIDE_MOV

    // Per-tile rigid body tracking (only used when state.rigid is true)
    std::vector<BitVec> rigidGroupIndexMask;
    std::vector<BitVec> rigidMovementAppliedMask;

    // Command queue
    std::vector<std::string> commandQueue;
    std::vector<int> commandQueueSourceRules;  // indices into ruleGroups

    // Row/col content masks for fast rule filtering
    std::vector<BitVec> colCellContents;
    std::vector<BitVec> rowCellContents;
    BitVec mapCellContents;

    std::vector<BitVec> colCellContents_Movements;
    std::vector<BitVec> rowCellContents_Movements;
    BitVec mapCellContents_Movements;

    Level() = default;

    Level(int lineNum, int w, int h, int lc, int strideObj, int strideMov)
        : lineNumber(lineNum), width(w), height(h), n_tiles(w * h),
          layerCount(lc), STRIDE_OBJ(strideObj), STRIDE_MOV(strideMov),
          objects(w * h * strideObj, 0),
          movements(w * h * strideMov, 0) {}

    void initMasks(bool rigid) {
        colCellContents.assign(width, BitVec(STRIDE_OBJ));
        rowCellContents.assign(height, BitVec(STRIDE_OBJ));
        mapCellContents = BitVec(STRIDE_OBJ);

        colCellContents_Movements.assign(width, BitVec(STRIDE_MOV));
        rowCellContents_Movements.assign(height, BitVec(STRIDE_MOV));
        mapCellContents_Movements = BitVec(STRIDE_MOV);

        movements.assign(n_tiles * STRIDE_MOV, 0);

        if (rigid) {
            rigidGroupIndexMask.assign(n_tiles, BitVec(STRIDE_MOV));
            rigidMovementAppliedMask.assign(n_tiles, BitVec(STRIDE_MOV));
        }
    }

    int delta_index(int direction) const {
        // direction bitmasks: up=1, down=2, left=4, right=8, action=16
        int dx = 0, dy = 0;
        switch (direction) {
            case 1:  dy = -1; break;  // up
            case 2:  dy = 1;  break;  // down
            case 4:  dx = -1; break;  // left
            case 8:  dx = 1;  break;  // right
            case 15: break;           // ?
            case 16: break;           // action
            case 3:  break;           // no
        }
        return dx * height + dy;
    }

    // Get cell as a BitVec (copy)
    BitVec getCell(int index) const {
        BitVec result(STRIDE_OBJ);
        int base = index * STRIDE_OBJ;
        for (int i = 0; i < STRIDE_OBJ; ++i)
            result.data[i] = objects[base + i];
        return result;
    }

    // Get cell into existing BitVec
    void getCellInto(int index, BitVec& target) const {
        int base = index * STRIDE_OBJ;
        for (int i = 0; i < STRIDE_OBJ; ++i)
            target.data[i] = objects[base + i];
    }

    void setCell(int index, const BitVec& vec) {
        int base = index * STRIDE_OBJ;
        for (int i = 0; i < STRIDE_OBJ; ++i)
            objects[base + i] = vec.data[i];
    }

    BitVec getMovements(int index) const {
        BitVec result(STRIDE_MOV);
        int base = index * STRIDE_MOV;
        for (int i = 0; i < STRIDE_MOV; ++i)
            result.data[i] = movements[base + i];
        return result;
    }

    void getMovementsInto(int index, BitVec& target) const {
        int base = index * STRIDE_MOV;
        for (int i = 0; i < STRIDE_MOV; ++i)
            target.data[i] = movements[base + i];
    }

    void setMovements(int index, const BitVec& vec) {
        int base = index * STRIDE_MOV;
        for (int i = 0; i < STRIDE_MOV; ++i)
            movements[base + i] = vec.data[i];

        int colIndex = index / height;
        int rowIndex = index % height;
        colCellContents_Movements[colIndex].ior(vec);
        rowCellContents_Movements[rowIndex].ior(vec);
        mapCellContents_Movements.ior(vec);
    }

    void clearMovementsRaw(int index) {
        int base = index * STRIDE_MOV;
        for (int i = 0; i < STRIDE_MOV; ++i)
            movements[base + i] = 0;
    }

    Level clone() const {
        Level c;
        c.lineNumber = lineNumber;
        c.width = width;
        c.height = height;
        c.n_tiles = n_tiles;
        c.layerCount = layerCount;
        c.STRIDE_OBJ = STRIDE_OBJ;
        c.STRIDE_MOV = STRIDE_MOV;
        c.objects = objects;
        return c;
    }

    // Recalculate all row/col/map content masks from scratch
    void calculateRowColMasks() {
        for (auto& v : colCellContents) v.setZero();
        for (auto& v : rowCellContents) v.setZero();
        mapCellContents.setZero();
        for (auto& v : colCellContents_Movements) v.setZero();
        for (auto& v : rowCellContents_Movements) v.setZero();
        mapCellContents_Movements.setZero();

        BitVec cellObj(STRIDE_OBJ);
        BitVec cellMov(STRIDE_MOV);

        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                int idx = x * height + y;
                getCellInto(idx, cellObj);
                colCellContents[x].ior(cellObj);
                rowCellContents[y].ior(cellObj);
                mapCellContents.ior(cellObj);

                getMovementsInto(idx, cellMov);
                colCellContents_Movements[x].ior(cellMov);
                rowCellContents_Movements[y].ior(cellMov);
                mapCellContents_Movements.ior(cellMov);
            }
        }
    }
};

// Backup of level state for undo
struct LevelBackup {
    std::vector<int32_t> dat;
    int width = 0;
    int height = 0;
};
