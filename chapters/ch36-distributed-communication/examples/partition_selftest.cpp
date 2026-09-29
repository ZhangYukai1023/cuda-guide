#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr int global_rows = 24;
int rows_for(int rank, int ranks) { return global_rows / ranks + (rank < global_rows % ranks ? 1 : 0); }
int first_row(int rank, int ranks) { return rank * (global_rows / ranks) + (rank < global_rows % ranks ? rank : global_rows % ranks); }
}

int main() {
    try {
        for (int ranks : {2, 3, 5, 7}) {
            std::vector<int> count(global_rows, 0);
            for (int rank = 0; rank < ranks; ++rank) {
                int first = first_row(rank, ranks), rows = rows_for(rank, ranks);
                if (rows < 1 || first < 0 || first + rows > global_rows) throw std::runtime_error("partition range invalid");
                for (int row = first; row < first + rows; ++row) ++count[row];
            }
            for (int occurrences : count) if (occurrences != 1) throw std::runtime_error("row missing or repeated");
            std::cout << "ranks=" << ranks << " rows covered exactly once: PASS\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "partition FAIL: " << e.what() << '\n';
        return 1;
    }
}
