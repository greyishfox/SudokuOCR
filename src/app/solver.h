#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <set>
#include <sstream>

class Solver
{
private:
    // Member variables
    const int N = 9;
    const int m_EMPTY = 0;

    /* ----------------------- Private member functions ----------------------- */
    bool rowChecker(std::vector<int> puzzle, const int row);
    bool colChecker(std::vector<int> puzzle, const int col);
    bool boxChecker(std::vector<int> puzzle, const int row, const int col);
    bool selectionChecker(std::vector<int> puzzle, const int row, const int col);
    bool findNextEmptyCell(std::vector<int> puzzle, int& row, int &col);

public:
    Solver();   // Constructor
    ~Solver();  // Destructor

    /* ----------------------- Public member functions ----------------------- */
    bool checker(const std::vector<int> puzzle, const int row, const int col);
    bool solve(std::vector<int>& puzzle, int row, int col);
    void printSudoku(std::vector<int> sudoku);
    std::vector<int> createSudokuPuzzle(const std::vector<bool> cellWithDigit, const std::string detectedDigits);
};

#endif // SOLVER_H
