#include "digraph.hh"
#include "routing_optimizer.hh"
#include "scalar_digraph.hh"
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>

void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name << " rows cols routing_type [--optimize] [--log filename] [--booksim filename]\n"
            << "  rows: Number of rows in the mesh (positive integer)\n"
            << "  cols: Number of columns in the mesh (positive integer)\n"
            << "  routing_type: Routing initialization type\n"
            << "    basic - Basic connectivity without turns\n"
            << "    xy - XY routing\n"
            << "    west - West-first routing\n"
            << "    oddeven - Odd-Even routing\n"
            << "  --optimize: Optional flag to apply routing optimizer\n"
            << "  --log filename: Optional flag to log turn edges to specified file\n"
            << "  --booksim filename: Optional flag to export network to BookSim2 anynet format\n";
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    print_usage(argv[0]);
    return 1;
  }

  // Parse arguments
  int rows = std::atoi(argv[1]);
  int cols = std::atoi(argv[2]);
  std::string routing_type = argv[3];
  bool optimize_flag = false;
  std::string log_filename = "";
  std::string booksim_filename = "";
  
  // Parse optional arguments
  for (int i = 4; i < argc; i++) {
    std::string arg = argv[i];
    
    if (arg == "--optimize") {
      optimize_flag = true;
    } 
    else if (arg == "--log" && i + 1 < argc) {
      log_filename = argv[i + 1];
      i++; // Skip the filename in the next iteration
    }
    else if (arg == "--booksim" && i + 1 < argc) {
      booksim_filename = argv[i + 1];
      i++; // Skip the filename in the next iteration
    }
    else {
      std::cout << "Error: Unknown argument '" << arg << "'\n";
      print_usage(argv[0]);
      return 1;
    }
  }
  
  digraph d(rows, cols); // Default initialization
  
  // Create digraph with selected routing type
  if (routing_type == "xy") {
    d = digraph::create_xy_routing(rows, cols);
    std::cout << "\nInitialized with XY routing:\n";
    
    // Debug: Check if all CPU paths exist (only if requested)
    if (!d.check_all_cpu_paths()) {
      std::cout << "Debugging missing CPU paths:\n";
      for (int src_x = 0; src_x < rows && src_x < 4; src_x++) {
        for (int src_y = 0; src_y < cols && src_y < 4; src_y++) {
          for (int dst_x = 0; dst_x < rows && dst_x < 4; dst_x++) {
            for (int dst_y = 0; dst_y < cols && dst_y < 4; dst_y++) {
              if (src_x == dst_x && src_y == dst_y)
                continue; // Skip same chip
              if (!d.has_cpu_path(src_x, src_y, dst_x, dst_y)) {
                std::cout << "Missing path: (" << src_x << "," << src_y 
                          << ") -> (" << dst_x << "," << dst_y << ")\n";
              }
            }
          }
        }
      }
      std::cout << "Debug complete (showing paths from 4x4 subset).\n";
    }
  } else if (routing_type == "west") {
    d = digraph::create_west_first_routing(rows, cols);
    std::cout << "\nInitialized with West-first routing:\n";
    
    // Debug: Check specifically which CPU paths are missing
    if (!d.check_all_cpu_paths()) {
      std::cout << "Debugging missing CPU paths:\n";
      int missing_count = 0;
      for (int src_x = 0; src_x < rows && missing_count < 10; src_x++) {
        for (int src_y = 0; src_y < cols && missing_count < 10; src_y++) {
          for (int dst_x = 0; dst_x < rows && missing_count < 10; dst_x++) {
            for (int dst_y = 0; dst_y < cols && missing_count < 10; dst_y++) {
              if (src_x == dst_x && src_y == dst_y)
                continue; // Skip same chip
              if (!d.has_cpu_path(src_x, src_y, dst_x, dst_y)) {
                std::cout << "Missing path: (" << src_x << "," << src_y 
                          << ") -> (" << dst_x << "," << dst_y << ")\n";
                missing_count++;
              }
            }
          }
        }
      }
      std::cout << "Debug complete (showing first " << missing_count << " missing paths).\n";
    }
  } else if (routing_type == "oddeven") {
    d = digraph::create_odd_even_routing(rows, cols);
    std::cout << "\nInitialized with Odd-Even routing:\n";
  } else if (routing_type == "basic") {
    // Default basic initialization
    std::cout << "\nInitialized with basic connectivity:\n";
  } else {
    std::cout << "Error: Unknown routing type '" << routing_type << "'\n";
    print_usage(argv[0]);
    return 1;
  }
  
  // Print initial state
  std::cout << "\nInitial state:\n";
  d.print_stats();
  
  // Apply optimization if requested
  if (optimize_flag) {
    std::cout << "\nApplying routing optimization...\n";
    
    // Set up random number generator and optimize
    std::random_device rd;
    std::mt19937_64 rng(rd());
    RoutingOptimizer opt(d);
    
    d.reset_perf_stats(); // Reset performance stats before optimization
    opt.reset_perf_stats();
    
    bool result = opt.optimize();
    
    std::cout << "\nAfter optimization:\n";
    d.print_stats();
    
    // Print performance statistics
    std::cout << "\nPerformance Statistics:\n";
    std::cout << "==========================================\n";
    opt.print_perf_stats();
    std::cout << "------------------------------------------\n";
    d.print_perf_stats();
    std::cout << "==========================================\n";
  }
  
  // Log turn edges if requested
  if (!log_filename.empty()) {
    std::cout << "\nLogging turn edges to file: " << log_filename << "\n";
    if (d.log_turn_edges(log_filename, routing_type, optimize_flag)) {
      std::cout << "Log file created successfully.\n";
    } else {
      std::cerr << "Error: Failed to create log file.\n";
    }
  }
  
  // Export to BookSim2 anynet format if requested
  if (!booksim_filename.empty()) {
    std::cout << "\nExporting network to BookSim2 anynet format: " << booksim_filename << "\n";
    if (d.export_booksim2_anynet(booksim_filename, routing_type, optimize_flag)) {
      std::cout << "BookSim2 configuration file created successfully.\n";
    } else {
      std::cerr << "Error: Failed to create BookSim2 configuration file.\n";
    }
  }
  
  std::cout << "Best Homogeneous: " << rows * cols * 6 << "\n";
  return 0;
}
