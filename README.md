# 🧬 Symbolic Regression Tool  

**Computational Intelligence – A.A. 2024/2025**  
📍 *Politecnico di Torino*  

---

## 📌 Project Overview  

This project implements a **symbolic regression tool** designed to approximate functions from continuous data and corresponding labels *(x, y)*.  

To achieve this, an **Evolutionary Algorithm (EA)** has been developed, refining expressions over multiple generations. More details about the algorithm structure and implementation can be found in the [Report](src/README.md).  

---

## 🔬 Evolutionary Algorithm Summary  

The **Evolutionary Algorithm (EA)** follows a **steady-state** model, where offspring are added to the population and then compete with parents for survival.  

Initially, a **generational model with an elitist strategy** was considered, but experimental results showed that the **steady-state model** yielded better performance.  

### 🛠️ **EA Configuration**  

- **Population Size**: 1000 (ensuring genetic diversity without excessive computational cost).  
- **Generations**: 50 (allowing sufficient evolution while maintaining reasonable execution time).  
- **Crossover Probability**: 70% (primary genetic operator).  
- **Mutation Probability**: 30% (applied when crossover is not performed).  

Empirical tests showed that the **algorithm preserves diversity** and continues improving fitness throughout the generations. A longer execution would likely lead to even better results.  

---

## 🔄 Genetic Algorithm Workflow  

1. **Selection of a genetic operator** (crossover or mutation) based on predefined probabilities.  
2. **Parent Selection**:  
   - **Tournament selection** with a fitness hole strategy.  
   - A subset of **τ = 10** individuals is randomly chosen from the population.  
   - With **90% probability**, the best among them is selected.  
   - In **10% of cases**, the least fit individual is selected to encourage genetic diversity.  
3. **Offspring Generation**:  
   - If **crossover** is selected, two parents generate new individuals.  
   - If **mutation** is chosen, a single individual is modified.  
4. **Survival Competition**:  
   - The offspring are merged with the parent population.  
   - The population is sorted by fitness.  
   - The top **population_size** individuals are retained for the next generation.  

### 🎯 **Why Fitness Holes?**  

Implementing a **fitness hole strategy** helps navigate complex adaptive landscapes. In traditional selection methods, intermediate steps in evolutionary pathways might be discarded due to temporary fitness reductions.  

By occasionally allowing **less fit individuals** to pass through, the algorithm can **overcome local optima** and explore new, potentially better solutions that require intermediate transformations.  

---

## 📑 Further Details  

For an in-depth explanation of the algorithm, implementation details, and experimental results, refer to the [Report](src/README.md).  
