# Y-DNA: Self-Evolving Branching Neural Networks

A genetic algorithm that evolves its own neural network architecture.  
Using tree-structured DNA, ***without backpropagation !!***, no human-designed topology.

![Status](https://img.shields.io/badge/status-research_prototype-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![Dependencies](https://img.shields.io/badge/dependencies-numpy_only-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

## The Core Idea

Current neural networks have a fundamental limitation: **humans design the architecture, then gradient descent optimizes the weights.** The topology is fixed before training begins — the number of layers, their connections, whether the network is recurrent or feedforward. These are all human decisions.

**Y-DNA removes the human from architecture design entirely.** Biology didnt require people to create brains either.

A genetic algorithm evolves tree-structured "DNA" that encodes branching neural network topologies. The DNA defines Y-shaped splits (one signal path becomes two), inverted-Y merges (two paths recombine), temporal memory nodes, addressable buffers, and cross-branch connections. The GA evolves *everything* — the topology, the weights, the memory structure, the connections between branches. No backpropagation. No gradient descent. No human deciding "use 12 layers with attention heads."

The architecture emerges from the problem. The GA discovers what structure the data needs.

## What Makes This Different

| Approach | Topology | Weights | Memory | Architecture Decisions |
|----------|----------|---------|--------|----------------------|
| Traditional NN | Fixed by human | Backpropagation | Fixed (LSTM/GRU or none) | Human |
| NEAT (2002) | Evolves nodes/connections | Evolution | No temporal memory | GA, but flat graph |
| NAS | Search over blocks | Backpropagation | Fixed | Automated but constrained |
| **Y-DNA** | **Evolves branching trees** | **Evolution (no backprop)** | **DELAY, PUSH/RECALL/POP** | **GA decides everything** |

### Key innovations:

- **Tree-structured DNA**: The genome is a branching tree of gene types, not a flat graph. Y-splits decompose problems; inverted-Y merges recombine. This naturally creates hierarchical processing.

- **Eight evolvable gene types**: NODE (processing), SPLIT (Y-branch), MERGE (inverted-Y), LOOP (recurrent memory), DELAY (fixed temporal lookback), PUSH (write to named buffer), RECALL (non-destructive read at position N), POP (destructive stack read). All share the same float-in/float-out interface. The GA decides which types to use where.

- **Cross-tree links**: Nodes can connect to any other node in the tree — siblings, parents, nodes in other input branches. Since there's no backpropagation, there's no mathematical constraint to stay feedforward. The GA evolves lateral and backward connections freely.

- **Addressable memory buffers**: PUSH/RECALL/POP operate on named buffer channels. Multiple branches can write and read from shared memory. RECALL can access any position (not just the top), enabling random-access temporal memory. This is the functional abstraction of what spiking neural networks achieve biologically.

- **Tiered population management**: Elites are protected and cached. Young chromosomes get a grace period after structural mutations. Established non-performers get early termination and are replaced with fresh random candidates. Wild survivors can mate with elites, grafting novel structures onto proven architectures.

- **No backpropagation**: The entire system — topology, weights, memory structure, cross-links — is evolved by the GA. This means the network can have arbitrary recurrent connections, cycles, and dynamic memory without worrying about gradient computation through complex graphs.

> Before you wonder why this network has such gene nodes with somewhat higher functions.
> The networks are not layered, and those functions were created in essence to perform somewhat akin to spiking neurons.
> Normally, a spiking neuron cannot work together with traditional neurons in an NN, but I wanted to have such functionality.
> So I added another way of what spiking neural networks essentially do.
> It should be able to solve any given problem, which you will see once you run it.
> v4 was rewritten to be multi-threaded, so far there is no GPU version as i dont depend on backprop of layers here...

## Architecture

```
            Input A          Input B          Input C
              |                |                |
          [Gene Tree]     [Gene Tree]      [Gene Tree]
           /     \            |             /     \
        SPLIT   DELAY       LOOP        NODE    PUSH
        /   \     |          |            |       |
     NODE  NODE  NODE    [memory]      RECALL   NODE
        \   /                              \   /
         \ /                                \ /
      [cross-link] ------>-----------> [cross-link]
              \                          /
               \                        /
                -----> MERGE <---------
                         |
                       OUTPUT
```

Each input gets its own gene tree. Trees can have arbitrary depth and branching.
Cross-tree links allow lateral communication between any nodes.
The output MERGE combines all branches.
The GA evolves all of this from a minimal seed.

## Results

The system is tested on 20 problems of increasing complexity:

### Logic & Decomposition
| Problem | Description | What it tests |
|---------|-------------|---------------|
| AND | Simple conjunction | Minimal — direct connections suffice |
| XOR | Exclusive or | Requires decomposition (Y-split) |
| A>B Compare | Compare two continuous values | Branch comparison + cross-links |
| (A XOR B) AND C | Hierarchical logic | Deep Y-branching |
| 3-bit Parity | Even/odd bit count | Multi-level decomposition |
| 4-bit Parity | Harder parity | Deeper decomposition |

### Temporal
| Problem | Description | What it tests |
|---------|-------------|---------------|
| Temporal Delay | Output = previous input | DELAY gene |
| Temporal XOR | XOR current with previous | LOOP + decomposition + cross-links |
| Echo@3 | Output = input from 3 steps ago | Deeper DELAY or RECALL |
| Sequence Copy | Store 4 values, then reproduce them | PUSH + RECALL |
| Sequence Reverse | Store 4 values, output reversed | PUSH + POP (natural stack behavior) |
| Pattern Match@2 | Detect when input matches 2 steps ago | RECALL + comparison |

### Arithmetic & String-like
| Problem | Description | What it tests |
|---------|-------------|---------------|
| Binary Addition | Full adder with carry (2 outputs) | Multi-output decomposition |
| Counting 1s | Running count of ones in stream | Temporal accumulation |
| Moving Average | Average of last 3 inputs | DELAY + arithmetic combination |
| Character Mapping | Nonlinear input-output lookup table | Function approximation |
| Mirror Detection | Detect palindrome in 6-step sequence | PUSH first half, compare reversed |
| ABA Grammar | Detect A_A pattern (first = third) | Memory + comparison |
| Run-Length Encoding | Count consecutive identical values | Temporal pattern + accumulation |
| Is-Sorted | Check if 3 inputs are in ascending order | Multi-branch comparison |

### Key observations from experiments:

- **The GA independently discovers which gene types each problem needs.** Non-temporal problems use NODE and SPLIT. Temporal problems grow LOOP and DELAY genes. Memory problems evolve PUSH/RECALL. Nobody tells the GA — the fitness landscape selects it.

- **Cross-tree links are transformative.** Problems that took 50+ generations without cross-links solve in 1-3 generations with them. The ability for branches to communicate laterally is a fundamental capability, not an optimization.

- **The DELAY gene instantly solves temporal delay problems** that previously stalled for 500+ generations with only LOOP genes. The right abstraction matters more than more computation.

- **Y-branching solves XOR faster than flat architectures.** The split/merge structure naturally decomposes non-linear problems. XOR solved in 1-17 generations vs 190 with flat block-based networks.

## Usage

```bash
# Run with default settings (200 population, 500 gens per level, 20 levels)
python ydna_v3.py

# For serious runs on a powerful machine, edit __main__:
# pop_size=400, max_gen_per_level=800
```

**Requirements:** Python 3.8+, NumPy. No other dependencies.

The output shows real-time evolution progress and ends with a summary table:

```
================================================================================
  Y-DNA v3 EVOLUTION SUMMARY
================================================================================
   Lv  Problem              Solved   Gen      Fit Genes Links Type  Architecture
--------------------------------------------------------------------------------
    0  AND                     YES     5  +0.9306     4     0      N:2 D:1
    1  XOR                     YES     1  +0.8591     4     1      N:1 L:1 D:1
    2  A>B Compare             YES     3  +0.8971     7     2      N:3 S:1 D:2
    3  Temporal Delay          YES     2  +0.8001     2     0 [T]  D:1
    ...
--------------------------------------------------------------------------------
  TOTAL: X/20 problems solved
================================================================================
```
Notice that not all problems may be solved (more generations may be required or a different seed).
The point is though it tries it evolves to many problems, i'm wokring on improvements but so far this is a game changer i think.
If you have real exotic hardware what would this able to do, so i encourage Anthropic or Google try it

## The Bigger Picture

This prototype demonstrates that **architecture can emerge from evolution rather than human design.** The branching Y-structure isn't just a topology — it's the fundamental operation of decomposition and recomposition, which is what all problem-solving is.

The system starts from a minimal seed (one node per input, directly connected to output) and grows whatever structure the problem demands. Logic problems get splits. Temporal problems get delays and loops. Memory problems get buffers. The architecture *is* the understanding.

### Connections to neuroscience

- **Branching trees** mirror cortical column organization
- **Cross-tree links** parallel lateral connections between brain regions
- **PUSH/RECALL/POP** abstract what spiking neural networks achieve biologically — charge accumulation and delayed firing
- **No backpropagation** aligns with biological learning: brains don't compute global error gradients

### Scaling considerations

This is a research prototype running single-threaded Python on toy problems. For real-world application:

- **Parallel evaluation**: Each chromosome is independent — trivially parallelizable across CPU cores
- **Compiled evaluation**: Converting gene trees to flat operation sequences would give 10-100x speedup
- **Island model**: Multiple sub-populations evolving independently with periodic migration
- **GPU acceleration**: Batch-evaluating chromosomes with similar topology structure

## Origin

This project emerged from a conversation between a developer with GA expertise (Peter Boos, [@PGTBoos](https://github.com/PGTBoos)) and Claude (Anthropic) about consciousness, and the essence of neural network architectures, and whether intelligence requires human-designed structure or can emerge from evolution alone like nature.

The core insight — that DNA should encode branching tree grammars (Y-splits and inverted-Y merges) rather than flat node/connection graphs — came from thinking about what a "layer" actually means in the context of the XOR problem, and generalizing from there to the idea that decomposition and recomposition are the primitive operations of understanding it. We could do away with layers and would not even require backprop. This is a first of its kind, in how we do it here, as far as we could research.

## License

MIT — use it, extend it, scale it up, prove us right or wrong.

## Contributing

This is early-stage research. If you're interested in:
- Scaling this to real-world problems
- GPU-accelerated evaluation
- Comparison benchmarks against NEAT, NAS, and other neuroevolution methods
- Theoretical analysis of the branching tree representation

Open an issue or reach out. The architecture is sound. It needs scale.
