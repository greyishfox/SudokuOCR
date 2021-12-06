#include "widget.h"
//#include "solver.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    // ------------------- Testing area ------------------------//
    // Terminal header
//    std::cout << "Sudoku Solver!" << std::endl;
//    Solver solver;
//    // Initial values to start the algorithm
//    int row = 0;
//    int col = 0;
//    // Example sudoku puzzle
//    std::vector<int> sudoku = {4,0,6,5,0,2,8,0,9,
//                               0,0,0,0,4,0,0,3,0,
//                               0,0,0,0,0,0,0,0,5,
//                               6,0,0,8,0,0,1,0,0,
//                               5,0,0,0,7,0,0,8,0,
//                               3,0,2,9,0,4,0,6,0,
//                               0,2,0,6,0,0,0,0,1,
//                               0,0,0,0,5,3,9,4,0,
//                               8,3,0,0,9,0,0,0,2};

//    auto start = std::chrono::high_resolution_clock::now();

//    // Perform the algorithm
//    if(solver.solve(sudoku, row, col))
//    {
//        if(solver.checker(sudoku, row, col))
//        {
//            solver.printSudoku(sudoku);
//            std::cout << "Congratulations, you solved the sudoku puzzle!" << std::endl;
//        }
//        else
//            std::cout << "Error: Backtracking leads to a wrong result" << std::endl;
//    }
//    else
//        std::cout << "Error: Sudoku cannot be solved" << std::endl;


//    auto finish = std::chrono::high_resolution_clock::now();

//    std::cout << "Time elapsed: " << (finish - start).count()*1e-9 << "s" << std::endl;
    // ------------------- Testing area ------------------------//
    QApplication a(argc, argv);
    Widget w;
    w.show();
    return a.exec();
}
