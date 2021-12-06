#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <set>

class Solver
{
private:
    // Member variables
    const int N = 9;
    const int m_empty = 0;

    // Member functions
    bool rowChecker(std::vector<int> puzzle, const int row);
    bool colChecker(std::vector<int> puzzle, const int col);
    bool boxChecker(std::vector<int> puzzle, const int row, const int col);
    bool selectionChecker(std::vector<int> puzzle, const int row, const int col);
    bool findNextEmptyCell(std::vector<int> puzzle, int& row, int &col);
public:
    // Solver(); // Note: no constructor needed
    bool checker(const std::vector<int> puzzle, const int row, const int col);
    bool solve(std::vector<int>& puzzle, int row, int col);
    void printSudoku(std::vector<int> sudoku);
};

#endif // SOLVER_H
