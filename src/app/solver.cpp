#include "solver.h"


// Checks validity of given row
bool Solver::rowChecker(std::vector<int> puzzle, const int row)
{
    // initialize flags with false;
    bool validNum_flag = false;
    bool uniqueNum_flag = false;
    std::vector<int>::iterator it = puzzle.begin() + row*N;

    // only check if valid row number
    if(row<N && row>=0)
    {
        // Test 1: Check if numbers are valid (1-9)  --> here we could also check the sum
        if(all_of(it, it+N, [this](int x){return (x>0 && x<=N);}))
            validNum_flag = true;

        // Test 2: Check if numbers are unique
        // sets only store unique values --> the size gives amount of unique values
        if((std::set<int>(it, it+N).size()) == N)
            uniqueNum_flag = true;
    }
    else
        std::cout << "Error: No valid row number!" << std::endl;

    // returns true if all rows contains unique and valid numbers
    return validNum_flag && uniqueNum_flag;
}

// Checks validity of given column
bool Solver::colChecker(std::vector<int> puzzle, const int col)
{
    // Initialize flags with false;
    bool validNum_flag = false;
    bool uniqueNum_flag = false;
    std::vector<int>::iterator it = puzzle.begin() + col;
    std::vector<int> validNum_vec;

    // Only check for valid row number
    if(col<N && col>=0)
    {
        // Test 1: Check if numbers are valid (1-9) --> here we could also check the sum
        for(; it < puzzle.end(); it += N)
            validNum_vec.push_back(*it);

        if(all_of(validNum_vec.begin(), validNum_vec.end(), [this](bool x){return x>0 && x<=N;}))
            validNum_flag = true;

        // Test 2: Check if numbers are unique
        it = validNum_vec.begin();
        if((std::set<int>(it, it+N).size()) == N)
            uniqueNum_flag = true;
    }
    else
        std::cout << "Error: No valid col number!" << std::endl;

    // Returns true if all rows contains unique and valid numbers
    return validNum_flag && uniqueNum_flag;
}

// Checks validity of given 3x3 box where the value at (row, col) is located
bool Solver::boxChecker(std::vector<int> puzzle, const int row, const int col)
{
    // initialize flags with false
    bool validNum_flag = false;
    bool uniqueNum_flag = false;

    // Only check for valid row number
    if(col<N && col>=0)
    {
        // Define vector for subsquare (3x3)
        std::vector<int> subSquare;

        // Get the first row and first column of the subsquare
        const int firstRow = (row/3)*3;
        const int firstCol = (col/3)*3;

        std::vector<int>::iterator it = puzzle.begin() + (firstRow*N + firstCol);

        // store values from subsquare in vector
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                subSquare.push_back(*(it+j));
            }
            it += N;
        }

        // Test 1: Check if numbers are valid (1-9) --> here we could also check the sum
        if(all_of(subSquare.begin(), subSquare.end(), [this](int x){return (x>0 && x<=N);}))
            validNum_flag = true;

        // Test 2: Check if numbers are unique
        if(std::set<int>(subSquare.begin(), subSquare.end()).size() == N)
            uniqueNum_flag = true;
    }

    // std::cout << "The square is: " << (validNum_flag && uniqueNum_flag) << std::endl;
    return validNum_flag && uniqueNum_flag;
}

// Validity bundle of rowChecker, colChecker and boxChecker
bool Solver::checker(const std::vector<int> puzzle, const int row, const int col)
{
    std::vector<bool> checkerFlags;
    std::set<int> rowSet;
    bool validity = false;
    checkerFlags.push_back(rowChecker(puzzle, row));
    checkerFlags.push_back(colChecker(puzzle, col));
    checkerFlags.push_back(boxChecker(puzzle, row, col));

    // Test 1: Check if numbers are valid (1-9) --> here we could also check the sum
    if(all_of(checkerFlags.begin(), checkerFlags.end(), [](bool flags){return flags==true;}))
        validity = true;

    return validity;
}

// Checks if inserted value is conform to sudoku rules
bool Solver::selectionChecker(std::vector<int> puzzle, const int row, const int col)
{
    /* 1) Check if number is unique in row */
    std::vector<int>::iterator it = puzzle.begin() + (row * N);
    std::vector<int>::iterator const selected = puzzle.begin() + (row * N + col);

    for(int i = 0; i < N; i++)
    {
        if(*it == *selected && it != selected)
            return false;
        ++it;
    }

    /* 2) Check if number is unique in col */
    it = puzzle.begin() + col;

    for(int i = 0; i < N; i++)
    {
        if(*it == *selected && it != selected)
            return false;
        it += N;
    }

    /* 3) Check if number is unique in box */
    // get the first row and first column of the subsquare
    const int firstRow = (row/3)*3;
    const int firstCol = (col/3)*3;

    it = puzzle.begin() + (firstRow*N + firstCol);

    // store values from subsquare in vector
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            if(*it == *selected && it != selected)
                return false;
            ++it;
        }
        it += (N-3);
    }
    return true;
}

// Finds the next unassigned ('0') cell
bool Solver::findNextEmptyCell(std::vector<int> puzzle, int &row, int &col)
{
    std::vector<int>::iterator it = puzzle.begin() + (row*N)+col;
    while(it < puzzle.end())
    {
        if(*it == m_empty)
        {
            row = std::distance(puzzle.begin(),it) / N;
            col = std::distance(puzzle.begin(),it) % N;
            return true;
        }
        else
            it++;
    }
    return false;
}

// Performs backtracking (common algorithm for sudoku solving)
bool Solver::solve(std::vector<int> &puzzle, int row, int col)
{
    /* 1) Check if last row and col is passed --> solved = true */
    if((row == (N-1)) && (col == N))
        return true;

    /* 2) if the last col is exceeded (== 9), jump to next row (++) and reset col (=0) */
    if(col == N)
    {
        row++;
        col = 0;
    }

    /* 3) Find the next unassigned cell (cell with a value of '0') */
    if(!findNextEmptyCell(puzzle, row, col))
        return true;

    // 4) Try numbers 1-9 for x and keep if the selection is valid
    std::vector<int>::iterator it;
    int index = 0;
    for(int x = 1; x <= N; x++)
    {
        index = (row * N) + col;
        puzzle[index] = x;
        if(selectionChecker(puzzle, row, col))
        {
            // Recursion: this part allows the backtracking
            if(solve(puzzle, row, col))
                return true;
            // Numbers that did not lead to a solution are set back to empty ('0')
            puzzle[index] = m_empty;
        }
        else
            puzzle[index] = m_empty;
    }
    // No suitable selection found
    return false;
}

void Solver::printSudoku(std::vector<int> sudoku)
{
    std::vector<int>::iterator itr = sudoku.begin();
    while(itr < sudoku.end())
    {
        copy(itr, itr+N, std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
        itr += N;
    }
}
