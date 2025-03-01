# CDG-BookSim2 Integration

This repository contains tools for integrating CDG (Constrained Directed Graph) routing with the BookSim2 network simulator.

## Overview

The CDG (Constrained Directed Graph) model explicitly represents turn restrictions in network routing. This allows for flexible routing algorithms while ensuring deadlock-freedom. This project enables:

1. Creation of various CDG-based routing algorithms (XY, West-first, Odd-even, Optimized)
2. Conversion of these CDG models to BookSim2 compatible format
3. Simulation of CDG-based routing in BookSim2

## Building

```bash
# Build all tools
make

# Build just the CDG-to-BookSim2 converter
make cdg_to_booksim

# Clean build files
make clean
```

## Usage

### Generating BookSim2 Configuration with CDG Routing

```bash
# Generate BookSim2 configuration for a 4x4 mesh with XY routing
./cdg_to_booksim 4 4 xy

# Generate West-first routing with a custom output prefix
./cdg_to_booksim 8 8 west_first custom_prefix

# Generate optimized routing (adds beneficial turns while maintaining deadlock freedom)
./cdg_to_booksim 4 4 optimized
```

This will generate several files:
- `{prefix}_anynet_file`: Network topology file for BookSim2
- `{prefix}_cdg_routing_table.txt`: CDG routing tables
- `{prefix}_booksim_config`: BookSim2 configuration file

### Running BookSim2 with CDG Routing

After building BookSim2 with our CDG routing enhancements:

```bash
cd path/to/booksim2-cdg
./booksim booksim_cdg_booksim_config
```

## CDG-based Routing Algorithms

The converter supports several routing algorithms:

1. **XY Routing**: Routes packets in X direction first, then Y direction
2. **West-first Routing**: Routes packets in westerly direction first if needed
3. **Odd-even Routing**: Uses different turn restrictions based on column parity
4. **Optimized Routing**: Starts with XY routing and adds beneficial turns while maintaining deadlock freedom

## How It Works

The integration works in several key steps:

1. **CDG Model**: We use the `digraph` class to model the network with turn restrictions
2. **Routing Generation**: Different routing functions create specific CDG configurations 
3. **BookSim2 Conversion**: The converter generates:
   - Network topology that matches the CDG connectivity
   - Routing tables that enforce turn restrictions
4. **Custom Routing Function**: BookSim2 has been enhanced with `min_cdg` and `adapt_cdg` routing functions that use the CDG routing tables

## Advanced Features

- **Unidirectional Channels**: Our modified BookSim2 supports true unidirectional channels
- **Adaptive Routing**: The `adapt_cdg` routing function provides congestion awareness
- **Routing Optimization**: The `RoutingOptimizer` class can add turns to improve performance while maintaining deadlock freedom

## Directory Structure

- `inc/`: Core CDG implementation
  - `digraph.hh`: CDG model implementation
  - `matrix.hh`: Bit matrix operations for CDG
  - `routing_optimizer.hh`: Tools for optimizing routing
- `src/`: Main source code
- `booksim2-cdg/`: Modified BookSim2 with CDG routing support
