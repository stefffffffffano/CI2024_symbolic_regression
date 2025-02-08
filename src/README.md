# Project-work: symbolic regression

The objective of the algorithm developed is, given a set of inputs x and the corresponding output y, to evolve a formula that best fits them. In order to do so, we developed an evolutionary algorithm that is described in the following.

The entire project has been done in collaboration with Andrea Mirenda. We worked together on his repo where I contributed, then I copied the results and the algorithm we produced together in a repo in my personal git.

**Genotype**

We decided to represent formulas through a tree with a maximum depth of 8 in order to avoid bloating, a phenomena that makes formulas grow generation after generation with small changes in fitness.

The tree is represented with a list of lists, where, depending on the number of operands taken by the operation, the structure is:

-`[operator, [first_operand], [second_operand]]` if it is a binary operation;

-`[operator, [operand]]` if it takes only one operand.

Constants and variables are simply represented through a string, e.g.: 'x[0]' and '9.48'.

**Phenotype**

The Phenotype is represented by a mathematical formula that can be evaluated with the corresponding input dimensions (1 or more in the problems that we had to face).

**Fitness**

The fitness evaluates the goodness of the solution. There were different possibilities in literature to represent the fitness of an individual in the context of symbolic regression. We started by using MSE as it is done to evaluate the performance of the delivered algorithm but, as soon as we started to execute different problems, we noticed that the scale changes a lot among them. Thus, it was difficult to compare the performance of the algorithms on different problems. We tried to substitute it with the MAPE, computed as:

```python

float(np.mean(np.abs((y - predictions) / y)) *100)

```

When using MAPE (Mean Absolute Percentage Error) and MSE (Mean Squared Error) as fitness measures in symbolic regression, discrepancies between these metrics were observed. MAPE measures the relative percentage error between predicted and actual values, making it highly sensitive to errors on samples with small actual values. In contrast, MSE measures the average squared error and is heavily influenced by errors on large values because the errors are squared.

In practice, this means that if your predictive formula is inaccurate on samples with large actual values, it could result in a low MSE (if errors are uniformly distributed across small and large values) while the MAPE could be high even if most of the relative errors occur on samples with small actual values.

This is the reason why, at the end, we decided to use the MSE as fitness and we didn't introduce other measures: our objective was exactly to minimize that value for the final evaluation. So, it doesn't matter if it has different values depending on the problem, it is simply related to the scale and the MAPE can be printed to have an idea of the performance of the algorithm. Moreover, given the aforementioned difference in scale, we decided to remove the penalization term that we had at the beginning in the fitness regarding the length of the formula. Our algorithm is already checking that the maximum depth is below 8 for every mutation and after the crossover, so we noticed no advantage in penalizing formulas based on the length.

On the other hand, we had to penalize programs with errors (a negative argument to the logarithm, for instance). We simply did it by setting the fitness to infinite in that case, in a way that the formula is discarded by the evolutionary process itself.

**Operations**

We have conducted several trials concerning the choice of operations. In this context, there are two possible approaches: including all potential operations or limiting the set to a smaller subset. The latter is supported by Taylor's theorem, which demonstrates that any function can be well approximated by a polynomial.

Our operation set, following the second strategy, includes: addition, subtraction, division, multiplication and power as binary operations, while we had sin,cos,exp log and sqrt as unary operations.

At the beginning we also tried to associate different probabilities to each operation, in a way that sum and subtraction were picked more with respect to sin and cosine, for example. However, we now consider all of them with the same probability because, after many trials, we have noticed that this is more effective to introduce more randomness.

**Individual**

Our individual is represented by a genome and the corresponding fitness. In order to do so, we used dataclass to ensure that the fitness is computed only once when the individual is generated or mutated and then used in different steps of the algorithm.

# Initial population

Before the EA algorithm effectively starts to run, the initial population has to be generated. In order to do so, we designed a function called `generate_program()` that takes as argument the input dimension. Then, for each dimension, it generates a random tree (calling `random_program()` ) for each dimension with a max_depth=2 and combines through randomly selected operations all the subtrees to generate the final tree.

This allows an initial population where there is a lot of genetic material related to all the dimensions of the problem and we can create more different combinations.

## Initial Population Generation

When we embarked on the development of our evolutionary algorithm for symbolic regression, we understood that the cornerstone of any successful evolutionary process is a well-conceived initial population. This is where our function, `random_program()`, plays a crucial role. Designed with recursion at its core, this function meticulously crafts each individual in the population as a unique expression tree.

### The Art of Recursive Tree Construction

The recursive nature of `random_program()` allows us to construct complex symbolic expressions in a controlled and systematic manner. Each call to the function has the potential to delve deeper into the tree, adding layers of mathematical operations and operands until a specified maximum depth is reached or a random condition prompts a halt.

-**Controlling Complexity with Depth**: The recursion depth parameter is a tool that we wield to balance the complexity of the expressions. By setting a maximum depth, we prevent the trees from becoming unwieldy, which is vital for maintaining the interpretability of the models and avoiding overfitting.

-**Deciding When to Stop**: At each recursive invocation, there's a 30% chance that the function will decide to generate a leaf node instead of continuing deeper. This stochastic element introduces variety in the depth across the population, ensuring that not all trees reach the maximum depth. It is a subtle yet powerful way to introduce randomness into the population, encouraging diversity.

-**The Genesis of Leaf Nodes**: When the recursion reaches its terminus—either because the maximum depth has been reached or the random chance triggers it—the function generates a leaf node. This node could be:

  -**A Variable**: If there are input variables that haven’t been used yet, and a subsequent 70% chance is met, a variable is selected. This choice is strategic, ensuring that each variable has a chance to contribute to the formula, thus exploiting the full informational potential of the input data.

  -**A Constant**: Constants are the unsung heroes in these trees, providing stability and adding an element of fixed numerical value that complements the dynamic nature of the variables.

A snapshot of the code is provided to provide all the needed details of th random_program function:

```python

defrandom_program(depth,input_dim,unary=False, used_indices=None):

    if used_indices isNone:

        used_indices = set()


    # Base case: generate a leaf node

    #If you already used all the variables, you can only use constants and return: don't place an operation between costants

    if depth ==0 or (random()<0.3) orlen(used_indices) == input_dim:

        iflen(used_indices) < input_dim andrandom()<0.7:

            available_indices = list(set(range(input_dim)) - used_indices)

            index = choice(available_indices)

            used_indices.add(index)

            returnf"x[{index}]", used_indices

        else: 

            #Only positive float between 0 and 10, no need to include negative ones, we have the subtraction operation

            returnstr(round(uniform(2, 10),2)), used_indices

     #Condition to check that we do not nest unary operations   

    if(not unary):

        operations = OPERATIONS

        weights = WEIGHTS

    else:

        operations = OPERATIONS_BINARY

        weights = BINARY_WEIGHTS

    op, arity, symbol,p = choices(operations,weights=weights, k=1)[0]

    if(unary !=True):

        unary = arity==1

    children = []

    for _ inrange(arity):

        child, used_indices = random_program(depth -1, input_dim, unary, used_indices)

        children.append(child)

    return [symbol] + children,used_indices

```

We have designed this function to have the following properties:

1.**Use of Input Dimensions:** The function uses a list, `used_indices`, to track which input dimensions (e.g., `0,1`, etc.) have been utilized in the formula. This ensures comprehensive coverage of all available input variables (if the depth makes it possible), making the formula relevant to all dimensions of the input data.

2.**Restrictions on Trigonometric Functions:** To enhance the mathematical sensibility of the formulas generated, the function includes a specific constraint regarding the nesting of trigonometric functions. Once a trigonometric function (such as `sin`, `cos`, etc.) is used, the function prohibits the inclusion of another trigonometric function within it. This constraint helps in preventing mathematically nonsensical expressions like `sin(cos(tan(x)))`, which, while computationally valid, may not be practically meaningful or may complicate the interpretation and analysis of the formula.

3.**Restrictions on operations with constants:** It is not allowed, in the current implementation, to add binary operations between constants. This is another kind of operation that makes the depth of the tree increase without the generation of a real meaningful formula.

This approach not only ensures that each formula is robust and contextually appropriate but also maintains clarity and reduces the computational redundancy that might arise from nested trigonometric operations. Such constraints are particularly important in scientific computing and simulations where the accuracy and interpretability of mathematical expressions are critical.

### Selecting the Right Operators and Operands

-**Operator Selection**: The decision on which mathematical operator to use is not taken lightly. Each operator adds a different dimension to the problem-solving capabilities of the tree. Whether it's a basic operation like addition or multiplication, or a more complex function like sine or logarithm, the choice is made randomly but with equal probability to maintain uniformity in operator distribution.

-**Recursive Operand Generation**: For nodes that are not leaves, the selected operator determines the number of operands. The function then recursively calls itself for each operand required, decrementing the depth with each call. This is where the true complexity of the tree is built, combining simplicity and complexity in a delicate dance that defines the nature of each individual expression tree.

### Why a Large Initial Population?

We typically start with a large initial population (1000 individuals). This breadth is not merely a number but a strategy to ensure that we cover as much of the solution space as possible right from the outset. It guards against the algorithm prematurely converging to suboptimal solutions and fosters robust evolutionary dynamics.

# Evolutionary algorithm

For what concerns the evolutionary algorithm, we decided to use a steady-state model where the offspring is added to the population, then they compete against parents for survival.

We also tried, at the beginning, to use a generational model with the elitist strategy, but this strategy was less effective than the chosen one.

We set all the parameters of the EA algorithm through empirical tests and, in particular, the population size is set to 1000, which ensures enough genetic material for the crossover without slowing too much each execution. The number of generations is set to 50 to obtain a result in a reasonable amount of time. We noticed, through experiments and collecting statistics about the population at each generation, that the diversity is preserved and the algorithm keeps evolving the formula and improving the fitness up to the last generation, so it would for sure benefit from a longer execution.

Starting from the initial population, parents are selected and the offspring is generated with two possible operations: mutation and crossover, that will be better explained in the following sections. The crossover is the main operation and it is executed with a 70% probability, while the remaining times mutation on a single individual is performed.

This is the hyper-modern GA flow:

- A genetic operator is selected with a given probability;
- the correct number of parents is selected;
- the offspring is generated.

When the offspirng has been generated, the population is extended with the offspring and they compete against each other for survival. After they have been merged together, they are sorted and only the best `population_size` individuals are kept for the next generation.

**Parent selection**

We implemented tournament selection with __fitness hole__ for selecting parents in order to generate the offspring. Starting from the initial population, tau=10 members are randomly selected and a tournament among them is performed. With a 90% probability, the best out of these tau individuals is returned. The remaining times, the less fit individual is returned.

Implementing a fitness hole as described can actually be beneficial in overcoming the challenges posed by an adaptive change that requires multiple intermediate steps. By intentionally allowing less fit individuals a chance to win, fitness holes can help navigate the evolutionary pathway where direct progression is hindered by intermediate steps that reduce overall fitness. This approach ensures that even though the final adaptation is advantageous, the evolutionary path to achieve it can successfully bypass 'fitness holes' that would otherwise deselect the intermediates before the final adaptation is achieved.

## Mutation

Before introducing the mutations and crossover techniques we used, we would like to acknowledge that we drew inspiration from the gplearn library. Although our implementation differs in several aspects, we aimed to incorporate hoist mutation and point mutation after recognizing their effectiveness. Regarding crossover, our approach deviates significantly, as our fitness function and tree structure differ from those used in gplearn.

We introduced three different types of mutation: subtree mutation, hoist mutation and point mutation.

- Subtree mutation: mutates a subtree with another one randomly generated;
- Hoist mutation: to replace a subtree with a smaller one within the selected subtree;
- Point mutation: to mutate a node (operator, variable or constant) of the program.

All of these functions are designed to achive a valid mutated solution.

Hoist and point mutation do not increase the depth of the solution (moreover Hoist mutation could be used to simplify the solutions), while subtree mutation can. To mantain the constraint on the max depth (set to 8) we defined a function called `cut_program` which simply cuts the tree if the depth is bigger than the max depth. To do so, we simply replace the subtree with a single number randomly chosen.

### Subtree mutation

The function begins by assigning the input `program` to `mutant`. This assignment assumes that a deep copy of `program` is unnecessary as it is managed outside the function, ensuring that the original program remains unaltered unless explicitly modified within this function.

#### Conditional Expansion

- The function first checks if the tree's depth is less than or equal to 4. If this condition is met and a randomly generated number is below 0.6, the tree is considered for expansion to increase its complexity:

  - A binary operation is randomly selected from `OPERATIONS_BINARY`, with the selection influenced by predefined weights (`BINARY_WEIGHTS`).
  - A new subtree is generated using `generate_program(input_dim)`, which creates a fresh symbolic expression based on the provided input dimension.
  - This new subtree is then combined with the original program under the selected binary operation, effectively creating a more complex tree structure.

#### Mutation Point Selection

- If the tree does not expand, the function identifies potential mutation points within the tree using `get_subtree_points_recursive(program)`. These points represent nodes in the tree where new subexpressions can be introduced.
- If no points are available (indicative of a simplistic program structure, possibly a single node), a completely new program is generated as a replacement.

#### Inserting New Subtrees

- A point for mutation is randomly selected from the available points.
- A new subtree is then created with `random_program(randint(1, max_depth), input_dim)[0]`, specifying a random depth within allowable limits to ensure diversity.
- This new subtree replaces the existing subtree at the chosen mutation point using `set_subtree_at_path(mutant, point, new_subtree)`, altering the program's structure.

#### Depth Control and Program Trimming

- After mutation, the depth of the mutant is checked. If it exceeds the maximum permissible depth (`MAX_TREE_DEPTH`), the program is trimmed using `cut_program(mutant)`. This function simplifies the tree by replacing deeper nodes with simpler constructs, such as constants, to maintain manageability and avoid overfitting.

This mutation function is designed to balance between introducing sufficient genetic diversity and maintaining the structural integrity and complexity of the symbolic expressions. It plays a vital role in the evolutionary dynamics by enabling adaptive changes that can lead to the discovery of more optimal solutions.

Notice that this is the only function for which an explicit call to the `cut_program` is needed to mantain valid the constraint on the depth of the tree.

```python

defmutate(program, input_dim, max_depth=2):

    """Mutation of a program."""

    mutant = program #the deep copy is done when passing the program when the function is called

    #If the program has a depth lower equal than 2, increase it

    if(depth(program)<=4 and random()<0.6):

        #randomly select a binary operation

        _, _, symbol,_ = choices(OPERATIONS_BINARY,weights=BINARY_WEIGHTS, k=1)[0]

        #generate a program

        new_subtree = generate_program(input_dim)

        #Combine the two programs

        return [symbol] + [program, new_subtree]

    points = get_subtree_points_recursive(program)

    ifnot points:

        returngenerate_program(input_dim)

  

    point = choice(points)

    new_subtree = random_program(randint(1,max_depth), input_dim)[0]


    mutant = set_subtree_at_path(mutant, point, new_subtree)

    if(depth(mutant)>MAX_TREE_DEPTH):

        mutant = cut_program(mutant)

    return mutant

```

### Hoist mutation

This function implements the hoist mutation within our evolutionary algorithm for symbolic regression. Hoist mutation is a method of simplifying the structure of a program tree by replacing it with one of its subtrees, effectively reducing its complexity and potentially improving its fitness by eliminating unnecessary parts.

#### Obtaining a Random Subtree

- The process begins by selecting a random subtree from the given program using the `get_subtree` function. This function also returns `parent_info`, which includes references to the parent of the subtree and the index where this subtree is attached, facilitating easy replacement.

#### Handling Leaf Nodes

- If the selected subtree is a leaf (i.e., not a list but a single node like a variable or a constant), the mutation cannot proceed because a leaf node does not contain any inner structures to promote to the higher level. In this case, the original program is returned unchanged, as hoist mutation is not applicable.

#### Replacing with Inner Subtree

- If the subtree is not a leaf, the function attempts to go deeper into the structure:

  - Another random subtree (`inner_subtree`) is selected from within the first subtree. This selection aims to find a deeper part of the tree that can be used to replace the higher-level structure.

#### Applying the Mutation

- Depending on the availability of `parent_info`:

  - If `parent_info` is `None`, it indicates that the selected subtree is the root of the program. In this case, the entire program is replaced with the `inner_subtree`. This can significantly simplify the program if the `inner_subtree` is substantially smaller than the original.
  - If `parent_info` is not `None`, the function retrieves the `parent` and the `idx` (index) from `parent_info`. The `inner_subtree` replaces its original position in the parent, thus hoisting the inner structure to a higher level in the overall tree.

This mutation method is particularly useful for reducing program complexity and avoiding bloat in genetic programming environments. By simplifying the program structure, hoist mutation can lead to more general solutions that may perform better on unseen data, contributing to the overall effectiveness of the evolutionary process.

```python

defhoist_mutation(program):

    """Mutation of a program through hoist"""

    # Obtein a random subtree

    subtree, parent_info = get_subtree(program)

  

    ifnotisinstance(subtree, list):

        # If the subtree is a leaf, return the program, hoist cannot be done

        return program


    # Get the inner subtree

    inner_subtree, _ = get_subtree(subtree)


    if parent_info isNone:

        # Sobstitute the program with the inner subtree

        return inner_subtree

    else:

        # Sobstitute the parent with the inner subtree

        parent, idx = parent_info

        parent[idx] = inner_subtree

        return program

```

### Point mutation

The `point_mutation` function is designed to introduce subtle changes at a single node within the symbolic expression tree, which represents a program. This type of mutation is fundamental for exploring slight variations in the genetic material, potentially leading to incremental improvements in the program's performance.

#### Overview of Functionality

-**Program**: The input to the function, represented as a nested list that reflects the tree structure of the symbolic program.

-**Max Tree Depth**: Specifies the maximum depth the tree can have, preventing the mutation from producing overly complex trees.

#### Mutation Process

The mutation process involves several helper functions, each tailored to handle specific parts of the mutation:

1.**Collecting Variables**:

   -`collect_variables(node, variables)`: This function recursively collects all the variable names used in the program, which are essential for potentially mutating variable nodes to other existing variables within the program.

2.**Mutating Leaf Nodes**:

   -`mutate_leaf(node)`: Targets leaf nodes (constants or variables) for mutation.

    - For variables (strings starting with 'x'), the function randomly selects a new variable from the set of all variables collected in the program.

    - For constants, it generates a new random constant within a specified range (-10 to 10, adjustable).

3.**Mutating Operator Nodes**:

   -`mutate_operator(node)`: Changes the operator at a node. It selects a new operator with the same arity (number of arguments) from the available operations, ensuring the structural integrity of the tree is maintained.

4.**Recursive Tree Traversal and Mutation Application**:

   -`recursive_mutation(node, current_depth)`: This is the core function that traverses the tree recursively.

    - If the current depth reaches the maximum allowed depth or the node is a leaf, it mutates the leaf node.

    - Otherwise, it randomly decides either to mutate the current operator node or to continue the recursion into one of the child nodes. This decision is based on a 30% chance to mutate the current node, with the rest leading to further recursive mutation.

#### Execution of Mutation

- The program tree is deeply copied at the beginning to preserve the original during mutation.
- The `recursive_mutation` function is then called with the root of the program tree, starting the mutation process from the top of the tree down to the leaves, applying changes as dictated by the tree structure and mutation rules.

#### Significance of Point Mutation

Point mutation is crucial for fine-tuning solutions within the population. By making small, targeted changes, it allows the evolutionary algorithm to explore the solution space around the current points, potentially discovering better performing solutions through minor adjustments. This mutation type is particularly effective for adjusting well-performing individuals that might only need slight modifications to optimize their performance.

This function, by varying the genetic structure of the programs slightly, ensures a continuous and diverse exploration of the solution space, which is essential for the success of the evolutionary process in symbolic regression.

```python

defpoint_mutation(program, max_tree_depth=MAX_TREE_DEPTH):

    """

    Perform a point mutation on the given program.


    Parameters:

    - program: list

        The symbolic program represented as a nested list (tree structure).

    - max_tree_depth: int

        Maximum depth allowed for the tree.


    Returns:

    - mutated_program: list

        A new program with a randomly mutated node.

    """


    #let's define a function that tell us the variables inside the program:

    defcollect_variables(node, variables):

        """Collect all variable names in the program."""

        ifisinstance(node, str) and node.startswith('x'):

            variables.add(node)

        elifisinstance(node, list):

            for child in node:

                collect_variables(child, variables)

      


    defmutate_leaf(node):

        """Mutate a leaf node (constant or variable)."""

        ifisinstance(node, str):

            # Variable mutation: Switch to another variable or a random constant that is present in 

            #the current program:

            if node.startswith('x'):

                # Collect variables from the program.

                variables = set()

                collect_variables(program, variables)

                returnchoice(list(variables))

            else:

                #Replace with a new random constant.

                #for now i consider only between -10 and 10, but we can change it

                returnstr(round(uniform(-10, 10), 2))

        return node


    defmutate_operator(node):

        """Mutate an operator node."""

        arity = len(node) -1

        #perations by matching arity

        valid_ops = [op for op inOPERATIONSif op[1]==arity]

  

        new_op = choice(valid_ops)

        node[0] = new_op[2]  # Replace operator symbol.


    defrecursive_mutation(node, current_depth):

        """Recursively traverse the tree and apply mutation."""

        if current_depth >= max_tree_depth ornotisinstance(node, list):

            # Mutate a leaf node if max depth reached or node is a leaf.

            returnmutate_leaf(node)


        ifisinstance(node, list):

            # Decide whether to mutate this operator or recurse further

            ifrandom() <0.3:  # 30% chance to mutate this operator but if yoi prefer we can change

                mutate_operator(node)

            else:

                # Recursively mutate a random child

                child_idx = randint(1, len(node) -1)

                node[child_idx] = recursive_mutation(node[child_idx], current_depth +1)


        return node


    mutated_program = copy.deepcopy(program)

    returnrecursive_mutation(mutated_program, current_depth=0)

```

## Crossover

The croossover function is designed to receive only 2 parents and, if one of them is a leaf program simply return casually one of the 2 programs (avoiding to perform the operation for programs with no childrens). Otherwise, select random indexes for both the parents and combine the first part of the tree with the second part of the tree of the 2 parents, returning a new individual.

In order to avoid bloating, the tree is cut if its depth is greater than MAX_TREE_DEPTH.

A snapshot of the code is provided to better describe the used aproach:

```python

defcrossover(parent1, parent2):

    """Crossover by subtree swapping."""

    #Obtain a casual subtree from parents

    subtree1, parent1_info = get_subtree(parent1)

    subtree2, _ = get_subtree(parent2)


    if parent1_info isNone:

        # A subtree of parent2 is returned, 

        # no risk of having more than MAX_TREE_DEPTH

        return subtree2

    else:

        # Subtree swapping and check on the final depth

        parent, idx = parent1_info

        parent[idx] = subtree2

        if(depth(parent)>MAX_TREE_DEPTH): #Cut the program only if necessary

            returncut_program(parent) 

        return parent

```

**Avoiding bloating**

In order to avoid bloating, we introduced a function called `cut_program` that recursively travels the tree that represents the program and substitutes one or more subtrees with a constant if the depth is too high after mutation or crossover.

We also tried many other techniques to avoid bloating, the main one has been to consider the depth of a solution directly inside the fitness function. But, since different program show different scales of MSE, it was difficult to reason a good and "general" penalty due to the "complexity" of the solution.

At the end, the best  seems to be to limit directly the depth of a program after any function that could have increased the depth.
