"""
Y-DNA v3: Self-Evolving Branching Neural Network
=================================================
DNA encodes a tree grammar of Y (split) and inverted-Y (merge) structures.
GA evolves both topology AND weights. No backpropagation.

Gene types (all float-in, float-out):
  NODE    - processing node with weight, bias, tanh
  SPLIT   - Y: one signal path becomes two branches
  MERGE   - inverted Y: combine two paths
  LOOP    - recurrent: blends current with previous output
  DELAY   - holds value for N steps, outputs the old one
  PUSH    - writes current value into named buffer
  RECALL  - reads from position N in named buffer (non-destructive)
  POP     - destructive read from top of named buffer

Cross-tree links: nodes can connect to siblings/parents outside
the normal tree flow via link_to references.

Author: Peter & Claude collaborative prototype v3
"""
import copy
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict
import numpy as np


# ============================================================
# DNA Gene Types
# ============================================================
class GeneType(Enum):
    NODE = "NODE"
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    LOOP = "LOOP"
    DELAY = "DELAY"
    PUSH = "PUSH"
    RECALL = "RECALL"
    POP = "POP"


# Shared buffers for PUSH/RECALL/POP (global per chromosome evaluation)
class BufferStore:
    """Named buffers shared across all genes in a chromosome."""
    def __init__(self):
        self.buffers: Dict[int, List[float]] = defaultdict(list)
        self.max_size = 16

    def push(self, channel: int, value: float):
        buf = self.buffers[channel]
        buf.append(value)
        if len(buf) > self.max_size:
            buf.pop(0)

    def recall(self, channel: int, position: int) -> float:
        buf = self.buffers[channel]
        if not buf:
            return 0.0
        idx = min(position, len(buf) - 1)
        return buf[-(idx + 1)]  # 0 = most recent

    def pop(self, channel: int) -> float:
        buf = self.buffers[channel]
        if not buf:
            return 0.0
        return buf.pop()  # LIFO: last in, first out

    def reset(self):
        self.buffers.clear()


@dataclass
class Gene:
    """A single gene in the DNA tree."""
    gene_type: GeneType
    weight: float = 1.0
    bias: float = 0.0
    left: Optional['Gene'] = None
    right: Optional['Gene'] = None
    depth: int = 0
    gene_id: int = 0
    # LOOP state
    memory: float = 0.0
    # MERGE mode (0=add, 1=multiply, 2=max)
    merge_mode: int = 0
    # DELAY: ring buffer
    delay_steps: int = 1           # how many steps to delay (GA evolves)
    delay_buffer: List[float] = field(default_factory=list)
    delay_pos: int = 0
    # PUSH/RECALL/POP: which named buffer channel
    buffer_channel: int = 0        # GA evolves which buffer to use
    # RECALL: which position to read from
    recall_position: int = 0       # 0=most recent, 1=one before, etc.
    # Cross-tree link: optional reference to another gene's output
    link_to: int = -1              # gene_id to pull signal from (-1 = none)
    link_weight: float = 0.0       # weight of the cross-link signal

    def count_nodes(self) -> int:
        c = 1
        if self.left:
            c += self.left.count_nodes()
        if self.right:
            c += self.right.count_nodes()
        return c

    def max_depth(self) -> int:
        ld = self.left.max_depth() if self.left else 0
        rd = self.right.max_depth() if self.right else 0
        return 1 + max(ld, rd)

    def count_by_type(self) -> Dict[str, int]:
        counts = {gt.value: 0 for gt in GeneType}
        counts[self.gene_type.value] += 1
        if self.left:
            for k, v in self.left.count_by_type().items():
                counts[k] += v
        if self.right:
            for k, v in self.right.count_by_type().items():
                counts[k] += v
        return counts

    def to_dict(self) -> dict:
        d = {
            "type": self.gene_type.value,
            "weight": round(self.weight, 3),
            "bias": round(self.bias, 3),
            "id": self.gene_id,
            "depth": self.depth,
        }
        if self.gene_type == GeneType.MERGE:
            d["merge_mode"] = ["add", "mul", "max"][self.merge_mode]
        if self.gene_type == GeneType.LOOP:
            d["memory"] = round(self.memory, 3)
        if self.gene_type == GeneType.DELAY:
            d["delay_steps"] = self.delay_steps
        if self.gene_type in (GeneType.PUSH, GeneType.RECALL, GeneType.POP):
            d["buffer_channel"] = self.buffer_channel
        if self.gene_type == GeneType.RECALL:
            d["recall_position"] = self.recall_position
        if self.link_to >= 0:
            d["link_to"] = self.link_to
            d["link_weight"] = round(self.link_weight, 3)
        if self.left:
            d["left"] = self.left.to_dict()
        if self.right:
            d["right"] = self.right.to_dict()
        return d


# ============================================================
# DNA Chromosome
# ============================================================
@dataclass
class Chromosome:
    input_trees: List[Gene] = field(default_factory=list)
    output_tree: Optional[Gene] = None
    n_inputs: int = 0
    n_outputs: int = 0
    fitness: float = 0.0
    generation_born: int = 0
    age: int = 0                   # generations since last structural mutation
    is_elite: bool = False         # protected from early termination
    _next_id: int = 0

    def next_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def total_genes(self) -> int:
        total = sum(t.count_nodes() for t in self.input_trees)
        if self.output_tree:
            total += self.output_tree.count_nodes()
        return total

    def all_gene_ids(self) -> List[int]:
        ids = []
        for tree in self.input_trees:
            _collect_ids(tree, ids)
        if self.output_tree:
            _collect_ids(self.output_tree, ids)
        return ids

    def describe(self) -> dict:
        return {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "total_genes": self.total_genes(),
            "max_depth": max((t.max_depth() for t in self.input_trees), default=0),
            "input_trees": [t.to_dict() for t in self.input_trees],
            "output_tree": self.output_tree.to_dict() if self.output_tree else None,
        }


def _collect_ids(gene: Gene, ids: List[int]):
    ids.append(gene.gene_id)
    if gene.left:
        _collect_ids(gene.left, ids)
    if gene.right:
        _collect_ids(gene.right, ids)


def _collect_genes_flat(gene: Gene) -> List[Gene]:
    result = [gene]
    if gene.left:
        result.extend(_collect_genes_flat(gene.left))
    if gene.right:
        result.extend(_collect_genes_flat(gene.right))
    return result


# ============================================================
# Chromosome Factory
# ============================================================
class ChromosomeFactory:
    @staticmethod
    def create_minimal(n_inputs: int, n_outputs: int) -> Chromosome:
        chrom = Chromosome(n_inputs=n_inputs, n_outputs=n_outputs)
        for i in range(n_inputs):
            node = Gene(
                gene_type=GeneType.NODE,
                weight=np.random.randn() * 0.5,
                bias=np.random.randn() * 0.1,
                gene_id=chrom.next_id(),
                depth=0
            )
            chrom.input_trees.append(node)
        if n_inputs > 1:
            chrom.output_tree = ChromosomeFactory._build_merge_chain(n_inputs, n_outputs, chrom)
        else:
            chrom.output_tree = Gene(
                gene_type=GeneType.NODE,
                weight=np.random.randn() * 0.5,
                bias=np.random.randn() * 0.1,
                gene_id=chrom.next_id(),
                depth=0
            )
        return chrom

    @staticmethod
    def _build_merge_chain(n_inputs: int, n_outputs: int, chrom: Chromosome) -> Gene:
        merge = Gene(
            gene_type=GeneType.MERGE, weight=1.0, bias=0.0,
            gene_id=chrom.next_id(), merge_mode=0, depth=0
        )
        out = Gene(
            gene_type=GeneType.NODE,
            weight=np.random.randn() * 0.5, bias=np.random.randn() * 0.1,
            gene_id=chrom.next_id(), depth=1
        )
        merge.left = out
        return merge


# ============================================================
# Forward Pass
# ============================================================
class Phenotype:
    @staticmethod
    def forward(chrom: Chromosome, inputs: np.ndarray,
                buffer_store: BufferStore, gene_outputs: Dict[int, float]) -> np.ndarray:
        branch_outputs = []
        for i, tree in enumerate(chrom.input_trees):
            val = float(inputs[i]) if i < len(inputs) else 0.0
            result = Phenotype._evaluate_gene(tree, val, buffer_store, gene_outputs)
            branch_outputs.append(result)

        if chrom.output_tree:
            merged = Phenotype._merge_branches(chrom.output_tree, branch_outputs,
                                                buffer_store, gene_outputs)
        else:
            merged = sum(branch_outputs) / max(len(branch_outputs), 1)

        if chrom.n_outputs == 1:
            return np.array([np.tanh(merged)])
        else:
            outputs = []
            for o in range(chrom.n_outputs):
                outputs.append(np.tanh(merged + o * 0.5))
            return np.array(outputs)

    @staticmethod
    def _evaluate_gene(gene: Gene, input_val: float,
                       buffer_store: BufferStore,
                       gene_outputs: Dict[int, float]) -> float:
        # Cross-tree link: add weighted signal from another gene
        if gene.link_to >= 0 and gene.link_to in gene_outputs:
            input_val += gene_outputs[gene.link_to] * gene.link_weight

        if gene.gene_type == GeneType.NODE:
            result = np.tanh(input_val * gene.weight + gene.bias)

        elif gene.gene_type == GeneType.SPLIT:
            left_val = Phenotype._evaluate_gene(gene.left, input_val, buffer_store, gene_outputs) if gene.left else input_val
            right_val = Phenotype._evaluate_gene(gene.right, input_val, buffer_store, gene_outputs) if gene.right else input_val
            result = left_val * gene.weight + right_val * (1.0 - gene.weight)

        elif gene.gene_type == GeneType.MERGE:
            if gene.left:
                result = Phenotype._evaluate_gene(gene.left, input_val, buffer_store, gene_outputs)
            else:
                result = np.tanh(input_val * gene.weight + gene.bias)

        elif gene.gene_type == GeneType.LOOP:
            combined = input_val * gene.weight + gene.memory * (1.0 - abs(gene.weight))
            gene.memory = np.tanh(combined + gene.bias)
            result = gene.memory
            if gene.left:
                result = Phenotype._evaluate_gene(gene.left, result, buffer_store, gene_outputs)

        elif gene.gene_type == GeneType.DELAY:
            # Ring buffer delay
            if len(gene.delay_buffer) < gene.delay_steps:
                # Fill up the buffer initially
                gene.delay_buffer.append(0.0)
                result = 0.0
            else:
                result = gene.delay_buffer[gene.delay_pos]
            # Write current value into buffer
            if len(gene.delay_buffer) < gene.delay_steps:
                gene.delay_buffer[-1] = input_val * gene.weight + gene.bias
            else:
                gene.delay_buffer[gene.delay_pos] = input_val * gene.weight + gene.bias
                gene.delay_pos = (gene.delay_pos + 1) % gene.delay_steps
            if gene.left:
                result = Phenotype._evaluate_gene(gene.left, result, buffer_store, gene_outputs)

        elif gene.gene_type == GeneType.PUSH:
            buffer_store.push(gene.buffer_channel, input_val * gene.weight + gene.bias)
            result = input_val  # pass through
            if gene.left:
                result = Phenotype._evaluate_gene(gene.left, result, buffer_store, gene_outputs)

        elif gene.gene_type == GeneType.RECALL:
            recalled = buffer_store.recall(gene.buffer_channel, gene.recall_position)
            # Blend recalled value with current input
            result = recalled * gene.weight + input_val * (1.0 - abs(gene.weight)) + gene.bias
            result = np.tanh(result)
            if gene.left:
                result = Phenotype._evaluate_gene(gene.left, result, buffer_store, gene_outputs)

        elif gene.gene_type == GeneType.POP:
            popped = buffer_store.pop(gene.buffer_channel)
            result = popped * gene.weight + gene.bias
            result = np.tanh(result)
            if gene.left:
                result = Phenotype._evaluate_gene(gene.left, result, buffer_store, gene_outputs)

        else:
            result = input_val

        # Store output for cross-tree links
        gene_outputs[gene.gene_id] = result
        return result

    @staticmethod
    def _merge_branches(output_tree: Gene, branch_values: List[float],
                        buffer_store: BufferStore,
                        gene_outputs: Dict[int, float]) -> float:
        if not branch_values:
            return 0.0
        if output_tree.gene_type == GeneType.MERGE:
            mode = output_tree.merge_mode
            if mode == 0:
                combined = sum(branch_values)
            elif mode == 1:
                combined = 1.0
                for v in branch_values:
                    combined *= v
            else:
                combined = max(branch_values, key=abs)
            if output_tree.left:
                combined = Phenotype._evaluate_gene(output_tree.left, combined,
                                                     buffer_store, gene_outputs)
            return combined
        else:
            combined = sum(branch_values)
            return Phenotype._evaluate_gene(output_tree, combined, buffer_store, gene_outputs)

    @staticmethod
    def reset_memory(chrom: Chromosome):
        for tree in chrom.input_trees:
            Phenotype._reset_gene_memory(tree)
        if chrom.output_tree:
            Phenotype._reset_gene_memory(chrom.output_tree)

    @staticmethod
    def _reset_gene_memory(gene: Gene):
        if gene.gene_type == GeneType.LOOP:
            gene.memory = 0.0
        if gene.gene_type == GeneType.DELAY:
            gene.delay_buffer = []
            gene.delay_pos = 0
        if gene.left:
            Phenotype._reset_gene_memory(gene.left)
        if gene.right:
            Phenotype._reset_gene_memory(gene.right)


# ============================================================
# Mutation Operators
# ============================================================
class Mutator:
    @staticmethod
    def mutate(chrom: Chromosome, generation: int, intensity: float = 1.0,
               stagnation: int = 0, level: int = 0) -> Chromosome:
        chrom = copy.deepcopy(chrom)

        if random.random() < 0.8 * intensity:
            Mutator._mutate_weights(chrom)

        if random.random() < 0.15 * intensity:
            Mutator._insert_split(chrom)

        if random.random() < 0.1 * intensity:
            Mutator._insert_node(chrom)

        if random.random() < 0.08 * intensity:
            Mutator._insert_loop(chrom)

        if random.random() < 0.06 * intensity:
            Mutator._insert_delay(chrom)

        # Pair mutations: these gene types are useless alone
        if random.random() < 0.06 * intensity:
            Mutator._insert_memory_pair(chrom)      # PUSH + RECALL

        if random.random() < 0.04 * intensity:
            Mutator._insert_stack_pair(chrom)        # PUSH + POP

        if random.random() < 0.06 * intensity:
            Mutator._insert_split_with_delay(chrom)  # SPLIT(NODE, DELAY)

        if random.random() < 0.05 * intensity:
            Mutator._insert_split_with_loop(chrom)   # SPLIT(NODE, LOOP)

        if random.random() < 0.04 * intensity:
            Mutator._add_cross_link(chrom)

        if random.random() < 0.05 * intensity:
            Mutator._remove_subtree(chrom)

        if random.random() < 0.1 * intensity:
            Mutator._change_merge_mode(chrom)

        if random.random() < 0.05 * intensity:
            Mutator._swap_branches(chrom)

        # Directed mutations for hard problems
        if stagnation > 30 and level >= 3:
            if random.random() < 0.25:
                Mutator._insert_split_with_loop(chrom)
            if random.random() < 0.25:
                Mutator._insert_split_with_delay(chrom)
            if random.random() < 0.3:
                Mutator._duplicate_subtree(chrom)
            if random.random() < 0.2:
                Mutator._insert_memory_pair(chrom)

        # Temporal problems: boost pair mutations
        if level in (3, 5, 6, 7, 8, 9, 13, 14, 16, 17, 18):
            if random.random() < 0.10 * intensity:
                Mutator._insert_split_with_delay(chrom)
            if random.random() < 0.08 * intensity:
                Mutator._insert_split_with_loop(chrom)
            if random.random() < 0.08 * intensity:
                Mutator._insert_memory_pair(chrom)
            if random.random() < 0.05 * intensity:
                Mutator._insert_stack_pair(chrom)

        return chrom

    @staticmethod
    def _mutate_weights(chrom: Chromosome):
        for tree in chrom.input_trees:
            Mutator._perturb_tree(tree)
        if chrom.output_tree:
            Mutator._perturb_tree(chrom.output_tree)

    @staticmethod
    def _perturb_tree(gene: Gene):
        if random.random() < 0.3:
            gene.weight += np.random.randn() * 0.3
            gene.weight = np.clip(gene.weight, -5.0, 5.0)
        if random.random() < 0.2:
            gene.bias += np.random.randn() * 0.2
            gene.bias = np.clip(gene.bias, -3.0, 3.0)
        # Type-specific mutations
        if gene.gene_type == GeneType.DELAY and random.random() < 0.15:
            gene.delay_steps = max(1, gene.delay_steps + random.choice([-1, 1]))
            gene.delay_steps = min(gene.delay_steps, 8)
        if gene.gene_type == GeneType.RECALL and random.random() < 0.2:
            gene.recall_position = max(0, gene.recall_position + random.choice([-1, 0, 1]))
            gene.recall_position = min(gene.recall_position, 15)
        if gene.gene_type in (GeneType.PUSH, GeneType.RECALL, GeneType.POP):
            if random.random() < 0.1:
                gene.buffer_channel = random.randint(0, 3)
        if gene.link_to >= 0 and random.random() < 0.2:
            gene.link_weight += np.random.randn() * 0.2
            gene.link_weight = np.clip(gene.link_weight, -3.0, 3.0)
        if gene.left:
            Mutator._perturb_tree(gene.left)
        if gene.right:
            Mutator._perturb_tree(gene.right)

    @staticmethod
    def _insert_split(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.count_nodes() < 25:
            new_node = Gene(
                gene_type=GeneType.NODE,
                weight=np.random.randn() * 0.5, bias=np.random.randn() * 0.1,
                gene_id=chrom.next_id(), depth=target.depth + 1,
            )
            old_copy = copy.deepcopy(target)
            target.gene_type = GeneType.SPLIT
            target.weight = random.random()
            target.gene_id = chrom.next_id()
            target.left = old_copy
            target.right = new_node

    @staticmethod
    def _insert_node(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.gene_type == GeneType.NODE and not target.left:
            target.left = Gene(
                gene_type=GeneType.NODE,
                weight=np.random.randn() * 0.5, bias=np.random.randn() * 0.1,
                gene_id=chrom.next_id(), depth=target.depth + 1,
            )

    @staticmethod
    def _insert_loop(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.gene_type == GeneType.NODE:
            target.gene_type = GeneType.LOOP
            target.memory = 0.0

    @staticmethod
    def _insert_delay(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.gene_type == GeneType.NODE:
            target.gene_type = GeneType.DELAY
            target.delay_steps = random.randint(1, 3)
            target.delay_buffer = []
            target.delay_pos = 0

    @staticmethod
    def _insert_push(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.gene_type == GeneType.NODE:
            target.gene_type = GeneType.PUSH
            target.buffer_channel = random.randint(0, 3)

    @staticmethod
    def _insert_recall(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.gene_type == GeneType.NODE:
            target.gene_type = GeneType.RECALL
            target.buffer_channel = random.randint(0, 3)
            target.recall_position = random.randint(0, 3)

    @staticmethod
    def _insert_pop(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.gene_type == GeneType.NODE:
            target.gene_type = GeneType.POP
            target.buffer_channel = random.randint(0, 3)

    @staticmethod
    def _add_cross_link(chrom: Chromosome):
        if not chrom.input_trees:
            return
        all_ids = chrom.all_gene_ids()
        if len(all_ids) < 2:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target:
            candidates = [gid for gid in all_ids if gid != target.gene_id]
            if candidates:
                target.link_to = random.choice(candidates)
                target.link_weight = np.random.randn() * 0.3

    @staticmethod
    def _remove_subtree(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.gene_type == GeneType.SPLIT:
            target.gene_type = GeneType.NODE
            target.weight = np.random.randn() * 0.5
            target.bias = np.random.randn() * 0.1
            target.left = None
            target.right = None

    @staticmethod
    def _change_merge_mode(chrom: Chromosome):
        if chrom.output_tree and chrom.output_tree.gene_type == GeneType.MERGE:
            chrom.output_tree.merge_mode = random.randint(0, 2)

    @staticmethod
    def _swap_branches(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.gene_type == GeneType.SPLIT and target.left and target.right:
            target.left, target.right = target.right, target.left

    @staticmethod
    def _duplicate_subtree(chrom: Chromosome):
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.left:
            target.right = copy.deepcopy(target.left)
            Mutator._perturb_tree(target.right)

    @staticmethod
    def _insert_split_with_delay(chrom: Chromosome):
        """Pair: SPLIT(NODE, DELAY) - current vs previous."""
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.count_nodes() < 20:
            current_node = Gene(
                gene_type=GeneType.NODE,
                weight=np.random.randn() * 0.5, bias=np.random.randn() * 0.1,
                gene_id=chrom.next_id(), depth=target.depth + 1,
            )
            delay_node = Gene(
                gene_type=GeneType.DELAY,
                weight=1.0, bias=0.0,
                delay_steps=random.randint(1, 3),
                delay_buffer=[], delay_pos=0,
                gene_id=chrom.next_id(), depth=target.depth + 1,
            )
            target.gene_type = GeneType.SPLIT
            target.weight = 0.5
            target.gene_id = chrom.next_id()
            target.left = current_node
            target.right = delay_node

    @staticmethod
    def _insert_split_with_loop(chrom: Chromosome):
        """Pair: SPLIT(NODE, LOOP) - immediate vs accumulated."""
        if not chrom.input_trees:
            return
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if target and target.count_nodes() < 20:
            current_node = Gene(
                gene_type=GeneType.NODE,
                weight=np.random.randn() * 0.5, bias=np.random.randn() * 0.1,
                gene_id=chrom.next_id(), depth=target.depth + 1,
            )
            loop_node = Gene(
                gene_type=GeneType.LOOP,
                weight=np.random.randn() * 0.5, bias=np.random.randn() * 0.1,
                memory=0.0,
                gene_id=chrom.next_id(), depth=target.depth + 1,
            )
            target.gene_type = GeneType.SPLIT
            target.weight = 0.5
            target.gene_id = chrom.next_id()
            target.left = current_node
            target.right = loop_node

    @staticmethod
    def _insert_memory_pair(chrom: Chromosome):
        """Pair: PUSH on one input tree, RECALL on another (or as child).
        Creates a store-and-retrieve channel."""
        if not chrom.input_trees:
            return
        channel = random.randint(0, 3)

        # Find a NODE to convert to PUSH
        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        push_target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if push_target and push_target.gene_type == GeneType.NODE:
            push_target.gene_type = GeneType.PUSH
            push_target.buffer_channel = channel

            # Now add a RECALL somewhere — prefer a different branch or as child
            recall_node = Gene(
                gene_type=GeneType.RECALL,
                weight=np.random.randn() * 0.5, bias=0.0,
                buffer_channel=channel,
                recall_position=random.randint(0, 3),
                gene_id=chrom.next_id(), depth=push_target.depth + 1,
            )
            # Attach as child of PUSH so data flows through
            if not push_target.left:
                push_target.left = recall_node
            else:
                # Or attach to a different tree if available
                other_trees = [i for i in range(len(chrom.input_trees)) if i != tree_idx]
                if other_trees:
                    other_idx = random.choice(other_trees)
                    other_target = Mutator._random_gene(chrom.input_trees[other_idx])
                    if other_target and other_target.gene_type == GeneType.NODE:
                        other_target.gene_type = GeneType.RECALL
                        other_target.buffer_channel = channel
                        other_target.recall_position = random.randint(0, 3)

    @staticmethod
    def _insert_stack_pair(chrom: Chromosome):
        """Pair: PUSH + POP on same channel. Natural stack behavior."""
        if not chrom.input_trees:
            return
        channel = random.randint(0, 3)

        tree_idx = random.randint(0, len(chrom.input_trees) - 1)
        push_target = Mutator._random_gene(chrom.input_trees[tree_idx])
        if push_target and push_target.gene_type == GeneType.NODE:
            push_target.gene_type = GeneType.PUSH
            push_target.buffer_channel = channel

            pop_node = Gene(
                gene_type=GeneType.POP,
                weight=np.random.randn() * 0.5, bias=0.0,
                buffer_channel=channel,
                gene_id=chrom.next_id(), depth=push_target.depth + 1,
            )
            if not push_target.left:
                push_target.left = pop_node
            else:
                other_trees = [i for i in range(len(chrom.input_trees)) if i != tree_idx]
                if other_trees:
                    other_idx = random.choice(other_trees)
                    other_target = Mutator._random_gene(chrom.input_trees[other_idx])
                    if other_target and other_target.gene_type == GeneType.NODE:
                        other_target.gene_type = GeneType.POP
                        other_target.buffer_channel = channel

    @staticmethod
    def _random_gene(gene: Gene) -> Optional[Gene]:
        genes = _collect_genes_flat(gene)
        return random.choice(genes) if genes else None


# ============================================================
# Crossover
# ============================================================
class Crossover:
    @staticmethod
    def cross(parent_a: Chromosome, parent_b: Chromosome, level: int = 0) -> Chromosome:
        child = copy.deepcopy(parent_a)
        if level >= 3 and parent_a.input_trees and parent_b.input_trees:
            for i in range(len(child.input_trees)):
                if i < len(parent_b.input_trees) and random.random() < 0.5:
                    donor_gene = Mutator._random_gene(parent_b.input_trees[i])
                    target_gene = Mutator._random_gene(child.input_trees[i])
                    if donor_gene and target_gene and target_gene.left:
                        target_gene.left = copy.deepcopy(donor_gene)
        else:
            for i in range(len(child.input_trees)):
                if i < len(parent_b.input_trees) and random.random() < 0.5:
                    child.input_trees[i] = copy.deepcopy(parent_b.input_trees[i])
        if parent_b.output_tree and random.random() < 0.3:
            child.output_tree = copy.deepcopy(parent_b.output_tree)
        return child


# ============================================================
# Problem Suite (expanded)
# ============================================================
class Problems:
    _cache: Dict[int, Tuple] = {}  # class-level cache

    @staticmethod
    def get_problem(level: int) -> Tuple[List[np.ndarray], List[np.ndarray], str, bool, int]:
        """Returns (inputs, targets, name, is_temporal, seq_length). Cached after first call."""
        if level in Problems._cache:
            return Problems._cache[level]

        problems = [
            Problems._and_gate,              # 0: trivial
            Problems._xor_gate,              # 1: needs decomposition
            Problems._two_feature_compare,   # 2: comparing branches
            Problems._temporal_delay,        # 3: output = prev input (DELAY/LOOP)
            Problems._pattern_hierarchy,     # 4: (A XOR B) AND C
            Problems._temporal_xor,          # 5: XOR with previous (LOOP + decomp)
            Problems._sequence_copy,         # 6: store then reproduce (PUSH/RECALL)
            Problems._sequence_reverse,      # 7: store then output backwards (PUSH/POP)
            Problems._echo_at_distance,      # 8: output = input from 3 steps ago (deeper DELAY)
            Problems._pattern_match,         # 9: output 1 when input matches N steps back (RECALL)
            Problems._parity3,               # 10: 3-bit parity
            Problems._parity4,               # 11: 4-bit parity
            # --- Arithmetic & string-like ---
            Problems._binary_addition,       # 12: 2-bit add with carry
            Problems._counting_ones,         # 13: running count of 1s in stream (accumulation)
            Problems._moving_average,        # 14: average of last 3 inputs
            Problems._char_mapping,          # 15: lookup table (input 0-3 maps to different output)
            Problems._mirror_detection,      # 16: detect if 2nd half mirrors 1st half
            Problems._simple_grammar,        # 17: detect ABA pattern in sequence
            Problems._run_length,            # 18: count consecutive identical inputs
            Problems._sequence_sorting,      # 19: input 3 values, output sorted order
        ]
        if level < len(problems):
            result = problems[level]()
        else:
            result = Problems._parity(min(level - 6, 6))

        Problems._cache[level] = result
        return result

    @staticmethod
    def _and_gate():
        ins = [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]
        outs = [np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0])]
        return ins, outs, "AND", False, 1

    @staticmethod
    def _xor_gate():
        ins = [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]
        outs = [np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([0.0])]
        return ins, outs, "XOR", False, 1

    @staticmethod
    def _two_feature_compare():
        ins, outs = [], []
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
                if a == b:
                    continue
                ins.append(np.array([a, b]))
                outs.append(np.array([1.0 if a > b else 0.0]))
        return ins, outs, "A>B Compare", False, 1

    @staticmethod
    def _temporal_delay():
        np.random.seed(999)
        ins, outs = [], []
        seq_length = 5
        for _ in range(8):
            seq = [float(np.random.randint(0, 2)) for _ in range(seq_length)]
            for t in range(len(seq)):
                ins.append(np.array([seq[t]]))
                prev = seq[t-1] if t > 0 else 0.0
                outs.append(np.array([prev]))
        np.random.seed(None)
        return ins, outs, "Temporal Delay", True, seq_length

    @staticmethod
    def _pattern_hierarchy():
        ins, outs = [], []
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    ins.append(np.array([float(a), float(b), float(c)]))
                    outs.append(np.array([float((a ^ b) & c)]))
        return ins, outs, "(A XOR B) AND C", False, 1

    @staticmethod
    def _temporal_xor():
        np.random.seed(888)
        ins, outs = [], []
        seq_length = 6
        for _ in range(8):
            seq = [float(np.random.randint(0, 2)) for _ in range(seq_length)]
            for t in range(len(seq)):
                ins.append(np.array([seq[t]]))
                result = float(int(seq[t]) ^ int(seq[t-1])) if t > 0 else seq[t]
                outs.append(np.array([result]))
        np.random.seed(None)
        return ins, outs, "Temporal XOR", True, seq_length

    @staticmethod
    def _sequence_copy():
        """Input a 4-step sequence, then 4 steps of zeros. Output should reproduce the sequence."""
        np.random.seed(777)
        ins, outs = [], []
        seq_length = 8  # 4 input + 4 recall
        for _ in range(6):
            seq = [float(np.random.randint(0, 2)) for _ in range(4)]
            # Phase 1: input the sequence, output 0 (just store)
            for t in range(4):
                ins.append(np.array([seq[t]]))
                outs.append(np.array([0.0]))
            # Phase 2: input zeros, output the stored sequence
            for t in range(4):
                ins.append(np.array([0.0]))
                outs.append(np.array([seq[t]]))
        np.random.seed(None)
        return ins, outs, "Seq Copy", True, seq_length

    @staticmethod
    def _sequence_reverse():
        """Input a 4-step sequence, then output it reversed. Natural for PUSH/POP."""
        np.random.seed(666)
        ins, outs = [], []
        seq_length = 8
        for _ in range(6):
            seq = [float(np.random.randint(0, 2)) for _ in range(4)]
            # Phase 1: input sequence, output 0
            for t in range(4):
                ins.append(np.array([seq[t]]))
                outs.append(np.array([0.0]))
            # Phase 2: output reversed
            for t in range(4):
                ins.append(np.array([0.0]))
                outs.append(np.array([seq[3 - t]]))
        np.random.seed(None)
        return ins, outs, "Seq Reverse", True, seq_length

    @staticmethod
    def _echo_at_distance():
        """Output equals input from 3 steps ago."""
        np.random.seed(555)
        ins, outs = [], []
        seq_length = 8
        for _ in range(6):
            seq = [float(np.random.randint(0, 2)) for _ in range(seq_length)]
            for t in range(seq_length):
                ins.append(np.array([seq[t]]))
                echo = seq[t-3] if t >= 3 else 0.0
                outs.append(np.array([echo]))
        np.random.seed(None)
        return ins, outs, "Echo@3", True, seq_length

    @staticmethod
    def _pattern_match():
        """Output 1 when current input matches input from 2 steps ago."""
        np.random.seed(444)
        ins, outs = [], []
        seq_length = 8
        for _ in range(6):
            seq = [float(np.random.randint(0, 2)) for _ in range(seq_length)]
            for t in range(seq_length):
                ins.append(np.array([seq[t]]))
                if t >= 2:
                    match = 1.0 if seq[t] == seq[t-2] else 0.0
                else:
                    match = 0.0
                outs.append(np.array([match]))
        np.random.seed(None)
        return ins, outs, "PatternMatch@2", True, seq_length

    @staticmethod
    def _parity3():
        return Problems._parity(3)

    @staticmethod
    def _parity4():
        return Problems._parity(4)

    # --- Arithmetic ---
    @staticmethod
    def _binary_addition():
        """Two 1-bit inputs + carry_in => sum, carry_out. Full adder."""
        ins, outs = [], []
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    s = (a + b + c) % 2
                    co = 1 if (a + b + c) >= 2 else 0
                    ins.append(np.array([float(a), float(b), float(c)]))
                    outs.append(np.array([float(s), float(co)]))
        return ins, outs, "Binary Add", False, 1

    @staticmethod
    def _counting_ones():
        """Input stream of 0/1, output running count of 1s (normalized)."""
        np.random.seed(333)
        ins, outs = [], []
        seq_length = 8
        for _ in range(6):
            seq = [float(np.random.randint(0, 2)) for _ in range(seq_length)]
            count = 0
            for t in range(seq_length):
                count += int(seq[t])
                ins.append(np.array([seq[t]]))
                # Normalize: count / max_possible so output stays in 0-1 range
                outs.append(np.array([count / seq_length]))
        np.random.seed(None)
        return ins, outs, "Count 1s", True, seq_length

    @staticmethod
    def _moving_average():
        """Output = average of last 3 inputs."""
        np.random.seed(222)
        ins, outs = [], []
        seq_length = 8
        for _ in range(6):
            seq = [float(np.random.randint(0, 2)) for _ in range(seq_length)]
            for t in range(seq_length):
                window = seq[max(0, t-2):t+1]
                avg = sum(window) / 3.0
                ins.append(np.array([seq[t]]))
                outs.append(np.array([avg]))
        np.random.seed(None)
        return ins, outs, "MovingAvg@3", True, seq_length

    # --- String-like / encoding ---
    @staticmethod
    def _char_mapping():
        """Lookup table: input 0=>1, 1=>0, 0.5=>0.5, 0.25=>0.75. Nonlinear mapping."""
        ins = [
            np.array([0.0]), np.array([0.25]), np.array([0.5]), np.array([0.75]), np.array([1.0]),
            np.array([0.0]), np.array([0.25]), np.array([0.5]), np.array([0.75]), np.array([1.0]),
        ]
        outs = [
            np.array([1.0]), np.array([0.75]), np.array([0.5]), np.array([0.25]), np.array([0.0]),
            np.array([1.0]), np.array([0.75]), np.array([0.5]), np.array([0.25]), np.array([0.0]),
        ]
        return ins, outs, "CharMap", False, 1

    @staticmethod
    def _mirror_detection():
        """6-step sequence: output 1 at step 6 if second half mirrors first half."""
        np.random.seed(111)
        ins, outs = [], []
        seq_length = 6
        for _ in range(8):
            first_half = [float(np.random.randint(0, 2)) for _ in range(3)]
            # Half the time make it a mirror, half not
            if np.random.random() < 0.5:
                second_half = list(reversed(first_half))
                is_mirror = 1.0
            else:
                second_half = [float(np.random.randint(0, 2)) for _ in range(3)]
                is_mirror = 1.0 if second_half == list(reversed(first_half)) else 0.0
            seq = first_half + second_half
            for t in range(seq_length):
                ins.append(np.array([seq[t]]))
                # Only output the answer at the last step
                if t == seq_length - 1:
                    outs.append(np.array([is_mirror]))
                else:
                    outs.append(np.array([0.0]))
        np.random.seed(None)
        return ins, outs, "Mirror Detect", True, seq_length

    @staticmethod
    def _simple_grammar():
        """Detect ABA pattern: 3-step input, output 1 if first == third."""
        np.random.seed(100)
        ins, outs = [], []
        seq_length = 3
        for _ in range(12):
            a = float(np.random.randint(0, 2))
            b = float(np.random.randint(0, 2))
            c = float(np.random.randint(0, 2))
            is_aba = 1.0 if a == c else 0.0
            for t, val in enumerate([a, b, c]):
                ins.append(np.array([val]))
                if t == 2:
                    outs.append(np.array([is_aba]))
                else:
                    outs.append(np.array([0.0]))
        np.random.seed(None)
        return ins, outs, "ABA Grammar", True, seq_length

    @staticmethod
    def _run_length():
        """Count consecutive identical values. Output the run length so far (normalized)."""
        np.random.seed(99)
        ins, outs = [], []
        seq_length = 8
        for _ in range(6):
            seq = [float(np.random.randint(0, 2)) for _ in range(seq_length)]
            run_len = 1
            for t in range(seq_length):
                if t > 0 and seq[t] == seq[t-1]:
                    run_len += 1
                elif t > 0:
                    run_len = 1
                ins.append(np.array([seq[t]]))
                outs.append(np.array([run_len / seq_length]))
        np.random.seed(None)
        return ins, outs, "RunLength", True, seq_length

    @staticmethod
    def _sequence_sorting():
        """3 inputs, output 1 if they are in ascending order, 0 otherwise."""
        ins, outs = [], []
        vals = [0.0, 0.25, 0.5, 0.75, 1.0]
        for a in vals:
            for b in vals:
                for c in vals:
                    if a == b or b == c or a == c:
                        continue
                    ins.append(np.array([a, b, c]))
                    is_sorted = 1.0 if a < b < c else 0.0
                    outs.append(np.array([is_sorted]))
        return ins, outs, "IsSorted3", False, 1

    @staticmethod
    def _parity(n: int):
        n = max(2, min(n, 6))
        ins, outs = [], []
        for i in range(2**n):
            bits = [(i >> b) & 1 for b in range(n)]
            ins.append(np.array(bits, dtype=float))
            outs.append(np.array([float(sum(bits) % 2)]))
        return ins, outs, f"{n}-bit Parity", False, 1

    @staticmethod
    def get_io_size(level: int) -> Tuple[int, int]:
        sizes = {
            0: (2,1), 1: (2,1), 2: (2,1),       # AND, XOR, A>B
            3: (1,1), 4: (3,1), 5: (1,1),        # Delay, Hierarchy, TempXOR
            6: (1,1), 7: (1,1), 8: (1,1),        # SeqCopy, SeqReverse, Echo
            9: (1,1), 10: (3,1), 11: (4,1),      # PatternMatch, Parity3, Parity4
            12: (3,2), 13: (1,1), 14: (1,1),     # BinaryAdd, Count1s, MovAvg
            15: (1,1), 16: (1,1), 17: (1,1),     # CharMap, Mirror, ABA
            18: (1,1), 19: (3,1),                 # RunLength, IsSorted
        }
        if level in sizes:
            return sizes[level]
        n = min(level - 6, 6)
        return max(n, 2), 1


# ============================================================
# Parallel evaluation worker (top-level for multiprocessing)
# ============================================================
def _evaluate_worker(chrom: Chromosome, level: int, allow_early_exit: bool) -> float:
    """Standalone evaluation function for parallel processing."""
    inputs, targets, name, is_temporal, seq_length = Problems.get_problem(level)
    total_error = 0.0
    range_step = seq_length if is_temporal else len(inputs)
    n_sequences = max(1, len(inputs) // range_step)
    early_check_at = max(1, n_sequences // 4) * range_step
    seq_count = 0

    for start_idx in range(0, len(inputs), range_step):
        Phenotype.reset_memory(chrom)
        buffer_store = BufferStore()
        gene_outputs: Dict[int, float] = {}

        end_idx = min(start_idx + range_step, len(inputs))
        for i in range(start_idx, end_idx):
            inp, target = inputs[i], targets[i]
            try:
                output = Phenotype.forward(chrom, inp, buffer_store, gene_outputs)
                output = np.clip(output, -1, 1)
                mapped_target = target * 2.0 - 1.0
                error = np.sum((output - mapped_target) ** 2)
                total_error += error
            except Exception:
                total_error += 10.0

        seq_count += range_step

        if allow_early_exit and seq_count >= early_check_at and seq_count < len(inputs):
            partial_max = seq_count * len(targets[0]) * 4.0
            partial_acc = 1.0 - (total_error / max(partial_max, 0.001))
            if partial_acc < 0.55:
                return partial_acc - chrom.total_genes() * 0.003

    max_error = len(inputs) * len(targets[0]) * 4.0
    accuracy = 1.0 - (total_error / max(max_error, 0.001))

    penalty_factor = 0.003 if level < 4 else 0.001
    complexity = chrom.total_genes() * penalty_factor

    if accuracy >= 0.99:
        accuracy += 0.05  # small bonus, capped below 1.1

    return accuracy - complexity


def _evaluate_batch_worker(batch: List[Tuple]) -> List[float]:
    """Evaluate a batch of chromosomes in one worker. Reduces IPC overhead."""
    results = []
    for chrom, level, allow_early in batch:
        results.append(_evaluate_worker(chrom, level, allow_early))
    return results


# ============================================================
# Evolution Engine
# ============================================================
class YDNAEvolution:
    def __init__(self, pop_size: int = 200, n_workers: int = 0):
        """
        n_workers: number of parallel workers for evaluation.
            0 = auto-detect CPU cores
            1 = sequential (no multiprocessing)
            N = use N worker processes
        """
        self.pop_size = pop_size
        self.population: List[Chromosome] = []
        self.generation = 0
        self.current_level = 0
        self.best_fitness_ever = -999
        self.stagnation = 0
        self.history: List[dict] = []
        # Parallel evaluation pool
        self._pool = None
        self._n_workers = n_workers
        if n_workers != 1:
            try:
                import multiprocessing as mp
                cores = n_workers if n_workers > 0 else max(1, mp.cpu_count() - 1)
                self._pool = mp.Pool(processes=cores)
                print(f"Parallel evaluation: {cores} workers")
            except Exception as e:
                print(f"Parallel init failed ({e}), using sequential")
                self._pool = None

    def __del__(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    def initialize(self, level: int = 0):
        self.current_level = level
        n_in, n_out = Problems.get_io_size(level)
        self.population = [
            ChromosomeFactory.create_minimal(n_in, n_out)
            for _ in range(self.pop_size)
        ]

    def evaluate(self, chrom: Chromosome, level: int, allow_early_exit: bool = False) -> float:
        inputs, targets, name, is_temporal, seq_length = Problems.get_problem(level)
        total_error = 0.0
        range_step = seq_length if is_temporal else len(inputs)
        n_sequences = max(1, len(inputs) // range_step)

        # Early termination: check after first 25% of sequences
        early_check_at = max(1, n_sequences // 4) * range_step
        seq_count = 0

        for start_idx in range(0, len(inputs), range_step):
            Phenotype.reset_memory(chrom)
            buffer_store = BufferStore()
            gene_outputs: Dict[int, float] = {}

            end_idx = min(start_idx + range_step, len(inputs))
            for i in range(start_idx, end_idx):
                inp, target = inputs[i], targets[i]
                try:
                    output = Phenotype.forward(chrom, inp, buffer_store, gene_outputs)
                    output = np.clip(output, -1, 1)
                    mapped_target = target * 2.0 - 1.0
                    error = np.sum((output - mapped_target) ** 2)
                    total_error += error
                except Exception:
                    total_error += 10.0

            seq_count += range_step

            # Early termination check: if hopeless after 25% of data, bail out
            if allow_early_exit and seq_count >= early_check_at and seq_count < len(inputs):
                partial_max = seq_count * len(targets[0]) * 4.0
                partial_acc = 1.0 - (total_error / max(partial_max, 0.001))
                if partial_acc < 0.55:  # way below solvable, don't waste time
                    # Extrapolate final fitness as low
                    max_error = len(inputs) * len(targets[0]) * 4.0
                    return partial_acc - chrom.total_genes() * 0.002

        max_error = len(inputs) * len(targets[0]) * 4.0
        accuracy = 1.0 - (total_error / max(max_error, 0.001))

        penalty_factor = 0.003 if level < 4 else 0.001
        complexity = chrom.total_genes() * penalty_factor

        if accuracy >= 0.99:
            accuracy += 0.05  # small bonus, capped below 1.1

        return accuracy - complexity

    def is_solved(self, chrom: Chromosome, level: int) -> bool:
        inputs, targets, _, is_temporal, seq_length = Problems.get_problem(level)
        range_step = seq_length if is_temporal else len(inputs)

        for start_idx in range(0, len(inputs), range_step):
            Phenotype.reset_memory(chrom)
            buffer_store = BufferStore()
            gene_outputs: Dict[int, float] = {}

            end_idx = min(start_idx + range_step, len(inputs))
            for i in range(start_idx, end_idx):
                inp, target = inputs[i], targets[i]
                try:
                    output = Phenotype.forward(chrom, inp, buffer_store, gene_outputs)
                    for o, t in zip(output, target):
                        if t > 0.5 and o < 0.3:
                            return False
                        if t < 0.5 and o > 0.3:
                            return False
                except Exception:
                    return False
        return True

    def run_generation(self) -> dict:
        n_elite = max(2, int(self.pop_size * 0.1))
        n_in, n_out = Problems.get_io_size(self.current_level)

        if self.best_fitness_ever > 0:
            early_kill_threshold = self.best_fitness_ever * 0.85
        else:
            early_kill_threshold = 0.55

        replacements_made = 0

        # --- Parallel evaluation ---
        # Split population into: cached elites (skip) and needs-evaluation
        eval_indices = []
        eval_chroms = []
        eval_early_flags = []

        for i, chrom in enumerate(self.population):
            is_elite_slot = (i < n_elite and chrom.is_elite)
            is_young = (chrom.age <= 2)

            if is_elite_slot and chrom.fitness > 0:
                continue  # cached

            allow_early = not is_elite_slot and not is_young and chrom.age > 3
            eval_indices.append(i)
            eval_chroms.append(chrom)
            eval_early_flags.append(allow_early)

        # Evaluate in parallel if pool available, otherwise sequential
        if self._pool is not None and len(eval_chroms) > 10:
            # Chunked: send batches to each worker, not individual chromosomes
            # This reduces pickling/IPC overhead dramatically
            n_workers = self._pool._processes
            chunk_size = max(1, len(eval_chroms) // n_workers)
            # Build chunks of (chrom, level, early_flag) tuples
            chunks = []
            for start in range(0, len(eval_chroms), chunk_size):
                end = min(start + chunk_size, len(eval_chroms))
                chunk = [(eval_chroms[j], self.current_level, eval_early_flags[j])
                         for j in range(start, end)]
                chunks.append(chunk)
            # Each worker evaluates a batch and returns list of fitnesses
            batch_results = self._pool.map(_evaluate_batch_worker, chunks)
            # Flatten results back to population
            flat_fitnesses = []
            for batch in batch_results:
                flat_fitnesses.extend(batch)
            for idx, fitness in zip(eval_indices, flat_fitnesses):
                self.population[idx].fitness = fitness
        else:
            # Sequential fallback
            for idx, chrom, early in zip(eval_indices, eval_chroms, eval_early_flags):
                chrom.fitness = self.evaluate(chrom, self.current_level, allow_early_exit=early)

        # Early termination replacements (sequential - modifies population)
        for i, chrom in enumerate(self.population):
            is_elite_slot = (i < n_elite and chrom.is_elite)
            is_young = (chrom.age <= 2)
            allow_early = not is_elite_slot and not is_young and chrom.age > 3

            if allow_early and chrom.fitness < early_kill_threshold:
                fresh = ChromosomeFactory.create_minimal(n_in, n_out)
                for _ in range(random.randint(1, 3)):
                    fresh = Mutator.mutate(fresh, self.generation, 1.5,
                                           stagnation=0, level=self.current_level)
                fresh.age = 0
                fresh.generation_born = self.generation
                fresh.fitness = self.evaluate(fresh, self.current_level, allow_early_exit=False)
                self.population[i] = fresh
                replacements_made += 1

        # Age all chromosomes
        for chrom in self.population:
            chrom.age += 1

        # Sort by fitness
        # Sort: best fitness first. At equal fitness, smallest wins (parsimony pressure)
        self.population.sort(key=lambda c: (round(c.fitness, 3), -c.total_genes()), reverse=True)
        best = self.population[0]
        solved = self.is_solved(best, self.current_level)

        # Mark elites
        for i, chrom in enumerate(self.population):
            chrom.is_elite = (i < n_elite)

        # Stagnation tracking
        if best.fitness > self.best_fitness_ever + 0.001:
            self.best_fitness_ever = best.fitness
            self.stagnation = 0
        else:
            self.stagnation += 1

        # Internal stagnation reset: diversify bottom half of population
        _, _, _, level_temporal, _ = Problems.get_problem(self.current_level)
        if self.stagnation > 0 and self.stagnation % 40 == 0:
            # Replace bottom 40% with mutated fresh chromosomes
            n_replace = int(self.pop_size * 0.4)
            for i in range(len(self.population) - n_replace, len(self.population)):
                fresh = ChromosomeFactory.create_minimal(n_in, n_out)
                for _ in range(random.randint(2, 5)):
                    fresh = Mutator.mutate(fresh, self.generation, 2.0,
                                           stagnation=self.stagnation, level=self.current_level)
                # Use pair mutations for temporal problems
                if level_temporal and random.random() < 0.5:
                    Mutator._insert_split_with_delay(fresh)
                if level_temporal and random.random() < 0.4:
                    Mutator._insert_split_with_loop(fresh)
                if level_temporal and random.random() < 0.3:
                    Mutator._insert_memory_pair(fresh)
                fresh.age = 0
                fresh.fitness = 0.0
                self.population[i] = fresh

        # Count gene types
        type_counts = {}
        for tree in best.input_trees:
            for k, v in tree.count_by_type().items():
                type_counts[k] = type_counts.get(k, 0) + v
        if best.output_tree:
            for k, v in best.output_tree.count_by_type().items():
                type_counts[k] = type_counts.get(k, 0) + v

        # Count cross-links
        n_links = 0
        for tree in best.input_trees:
            for g in _collect_genes_flat(tree):
                if g.link_to >= 0:
                    n_links += 1

        _, _, problem_name, is_temporal, _ = Problems.get_problem(self.current_level)

        record = {
            "generation": self.generation,
            "level": self.current_level,
            "problem": problem_name,
            "best_fitness": round(best.fitness, 4),
            "avg_fitness": round(np.mean([c.fitness for c in self.population]), 4),
            "total_genes": best.total_genes(),
            "max_depth": max((t.max_depth() for t in best.input_trees), default=0),
            "gene_types": type_counts,
            "cross_links": n_links,
            "solved": solved,
            "stagnation": self.stagnation,
            "is_temporal": is_temporal,
            "dna": best.describe()
        }
        self.history.append(record)

        # --- Tiered mating strategy ---
        elites = self.population[:n_elite]
        # Collect "wild survivors" = non-elite chromosomes that are young (recently injected)
        wilds = [c for c in self.population[n_elite:] if c.age <= 3 and c.fitness > 0.6]

        intensity = 1.0 + min(self.stagnation * 0.03, 1.5)
        # Elites survive with preserved age
        new_pop = []
        for e in elites:
            ec = copy.deepcopy(e)
            ec.age = e.age  # preserve age through copy
            new_pop.append(ec)

        while len(new_pop) < self.pop_size:
            roll = random.random()

            if roll < 0.45:
                # Elite x Elite mutation (refinement) - 45%
                parent = random.choice(elites)
                child = Mutator.mutate(parent, self.generation, intensity,
                                       stagnation=self.stagnation, level=self.current_level)
                child.age = 0
                child.generation_born = self.generation

            elif roll < 0.70:
                # Elite x Elite crossover + mutation - 25%
                p1 = random.choice(elites)
                p2 = random.choice(elites)
                child = Crossover.cross(p1, p2, level=self.current_level)
                child = Mutator.mutate(child, self.generation, intensity * 0.5,
                                       stagnation=self.stagnation, level=self.current_level)
                child.age = 0
                child.generation_born = self.generation

            elif roll < 0.85 and wilds:
                # Elite x Wild crossover (exploration) - 15% when wilds available
                p_elite = random.choice(elites)
                p_wild = random.choice(wilds)
                child = Crossover.cross(p_elite, p_wild, level=self.current_level)
                child = Mutator.mutate(child, self.generation, intensity * 0.7,
                                       stagnation=self.stagnation, level=self.current_level)
                child.age = 0
                child.generation_born = self.generation

            elif roll < 0.95:
                # Fresh random with heavy mutation (wild injection) - 10%
                child = ChromosomeFactory.create_minimal(n_in, n_out)
                for _ in range(random.randint(2, 5)):
                    child = Mutator.mutate(child, self.generation, 2.0,
                                           stagnation=0, level=self.current_level)
                child.age = 0
                child.generation_born = self.generation

            else:
                # Wild x Wild (pure chaos) - 5%
                if len(wilds) >= 2:
                    p1 = random.choice(wilds)
                    p2 = random.choice(wilds)
                    child = Crossover.cross(p1, p2, level=self.current_level)
                else:
                    child = ChromosomeFactory.create_minimal(n_in, n_out)
                child = Mutator.mutate(child, self.generation, 2.5,
                                       stagnation=self.stagnation, level=self.current_level)
                child.age = 0
                child.generation_born = self.generation

            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        return record

    def advance_level(self):
        self.current_level += 1
        n_in_new, n_out_new = Problems.get_io_size(self.current_level)
        n_in_old, n_out_old = Problems.get_io_size(self.current_level - 1)

        if n_in_new != n_in_old or n_out_new != n_out_old:
            self.population = [
                ChromosomeFactory.create_minimal(n_in_new, n_out_new)
                for _ in range(self.pop_size)
            ]
        else:
            top = self.population[:5]
            new_pop = []
            while len(new_pop) < self.pop_size:
                parent = random.choice(top)
                new_pop.append(Mutator.mutate(parent, self.generation, 1.5,
                                              stagnation=0, level=self.current_level))
            self.population = new_pop

        # Reset all fitness — cached values from previous level are invalid
        for chrom in self.population:
            chrom.fitness = 0.0
            chrom.is_elite = False

        self.best_fitness_ever = -999
        self.stagnation = 0


# ============================================================
# Main Runner (per-level budget)
# ============================================================
def run(max_gen_per_level: int = 500, max_level: int = 20, pop_size: int = 200, n_workers: int = 0):
    engine = YDNAEvolution(pop_size=pop_size, n_workers=n_workers)
    engine.initialize(level=0)
    print("Y-DNA v3: Self-Evolving Branching Neural Network")
    print("=" * 65)
    print("Gene types: NODE SPLIT MERGE LOOP DELAY PUSH RECALL POP")
    print("Features: cross-tree links, named buffers, no backprop")
    print("=" * 65)

    # Track results per level
    results_summary = []

    while engine.current_level < max_level:
        _, _, problem_name, is_temporal, _ = Problems.get_problem(engine.current_level)
        temp_tag = " [TEMPORAL]" if is_temporal else ""
        print(f"\n>>> Starting Level {engine.current_level}: {problem_name}{temp_tag} <<<")
        engine.generation = 0
        engine.best_fitness_ever = -999
        engine.stagnation = 0
        level_solved = False

        for gen in range(max_gen_per_level):
            record = engine.run_generation()

            if gen % 10 == 0 or record["solved"]:
                types = record["gene_types"]
                t = types
                type_str = (f"N:{t.get('NODE',0)} S:{t.get('SPLIT',0)} "
                            f"L:{t.get('LOOP',0)} D:{t.get('DELAY',0)} "
                            f"Pu:{t.get('PUSH',0)} Re:{t.get('RECALL',0)} "
                            f"Po:{t.get('POP',0)}")
                links = f"Lnk:{record['cross_links']}" if record['cross_links'] > 0 else ""
                status = " SOLVED!" if record["solved"] else ""
                temp = " [T]" if record["is_temporal"] else ""
                print(
                    f"Gen {record['generation']:4d} | L{record['level']:2d} "
                    f"{record['problem']:>16s}{temp} | "
                    f"Fit:{record['best_fitness']:+.4f} | "
                    f"G:{record['total_genes']:3d} | "
                    f"{type_str} {links}| "
                    f"Stag:{record['stagnation']}{status}"
                )

            if record["solved"]:
                print(f"\n>>> Level {engine.current_level} SOLVED at gen {gen}!")
                best = engine.population[0]
                for i, tree in enumerate(best.input_trees):
                    print(f"    Input {i}: depth={tree.max_depth()}, {tree.count_by_type()}")

                # Record summary
                best_types = record["gene_types"]
                results_summary.append({
                    "level": engine.current_level,
                    "problem": problem_name,
                    "solved": True,
                    "gen_solved": gen,
                    "fitness": record["best_fitness"],
                    "genes": record["total_genes"],
                    "gene_types": best_types,
                    "cross_links": record["cross_links"],
                    "temporal": is_temporal,
                })
                level_solved = True
                break

            if engine.stagnation > 60:
                print(f">>> Stagnation reset at gen {gen}")
                n_in, n_out = Problems.get_io_size(engine.current_level)
                _, _, _, level_temporal, _ = Problems.get_problem(engine.current_level)
                top3 = engine.population[:3]
                fresh = [ChromosomeFactory.create_minimal(n_in, n_out)
                         for _ in range(engine.pop_size - 3)]
                for f in fresh:
                    # Standard mutations
                    for _ in range(random.randint(1, 4)):
                        f = Mutator.mutate(f, engine.generation, 2.0,
                                           stagnation=0, level=engine.current_level)
                    # If temporal problem, force temporal genes into some fresh chromosomes
                    if level_temporal and random.random() < 0.5:
                        Mutator._insert_loop(f)
                        Mutator._insert_delay(f)
                    if level_temporal and random.random() < 0.3:
                        Mutator._insert_push(f)
                        Mutator._insert_recall(f)
                engine.population = [copy.deepcopy(t) for t in top3] + fresh
                engine.stagnation = 0
                engine.best_fitness_ever = -999

        if not level_solved:
            print(f"\n>>> Level {engine.current_level} NOT solved in {max_gen_per_level} gens.")
            print(">>> Advancing anyway...\n")
            # Record as unsolved
            last_record = engine.history[-1] if engine.history else {}
            results_summary.append({
                "level": engine.current_level,
                "problem": problem_name,
                "solved": False,
                "gen_solved": -1,
                "fitness": last_record.get("best_fitness", 0),
                "genes": last_record.get("total_genes", 0),
                "gene_types": last_record.get("gene_types", {}),
                "cross_links": last_record.get("cross_links", 0),
                "temporal": is_temporal,
            })

        # Save history
        slim = []
        for r in engine.history:
            entry = {k: v for k, v in r.items() if k != "dna"}
            if r["solved"] or r["generation"] % 50 == 0:
                entry["dna"] = r["dna"]
            slim.append(entry)
        with open("ydna_v3_history.json", "w") as f:
            json.dump(slim, f, indent=2)

        if engine.current_level < max_level - 1:
            engine.advance_level()
        else:
            print(">>> All levels complete!")
            break

    # ============================================================
    # Print Summary
    # ============================================================
    print("\n")
    print("=" * 80)
    print("  Y-DNA v3 EVOLUTION SUMMARY")
    print("=" * 80)
    print(f"  {'Lv':>3s}  {'Problem':<20s} {'Solved':>6s} {'Gen':>5s} {'Fit':>8s} "
          f"{'Genes':>5s} {'Links':>5s} {'Type':>4s}  Architecture")
    print("-" * 80)

    solved_count = 0
    total_count = len(results_summary)

    for r in results_summary:
        lv = r["level"]
        name = r["problem"]
        solved = "YES" if r["solved"] else "NO"
        gen = str(r["gen_solved"]) if r["solved"] else "---"
        fit = f"{r['fitness']:+.4f}"
        genes = str(r["genes"])
        links = str(r["cross_links"])
        temporal = "[T]" if r["temporal"] else "   "

        # Compact architecture description
        gt = r["gene_types"]
        arch_parts = []
        for key in ["NODE", "SPLIT", "MERGE", "LOOP", "DELAY", "PUSH", "RECALL", "POP"]:
            v = gt.get(key, 0)
            if v > 0:
                short = {"NODE": "N", "SPLIT": "S", "MERGE": "M", "LOOP": "L",
                         "DELAY": "D", "PUSH": "Pu", "RECALL": "Re", "POP": "Po"}[key]
                arch_parts.append(f"{short}:{v}")
        arch = " ".join(arch_parts)

        if r["solved"]:
            solved_count += 1

        print(f"  {lv:3d}  {name:<20s} {solved:>6s} {gen:>5s} {fit:>8s} "
              f"{genes:>5s} {links:>5s} {temporal}  {arch}")

    print("-" * 80)
    print(f"  TOTAL: {solved_count}/{total_count} problems solved")
    print("=" * 80)

    return engine


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    # n_workers: 0=auto-detect cores, 1=sequential, N=use N cores
    # For harder problems, per problem adjus max_gen_per_level  and pop_size
    engine = run(max_gen_per_level=1000, max_level=20, pop_size=500, n_workers=0)
