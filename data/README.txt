Files:

1. db_numprop-4_nestlim-100.db: An sqlite3 database with two tables, 'status' and 'data' (from command 'SELECT name FROM sqlite_master WHERE type='table';').
- 'status' says of each LoT whether all the formulas have been found. 
- 'data' contains the following columns (from command 'PRAGMA table_info(data);'):
 (0, 'O', 'INTEGER'),
 (1, 'A', 'INTEGER'),
 (2, 'N', 'INTEGER'),
 (3, 'C', 'INTEGER'),
 (4, 'B', 'INTEGER'),
 (5, 'X', 'INTEGER'),
 (6, 'NA', 'INTEGER'),
 (7, 'NOR', 'INTEGER'),
 (8, 'NC', 'INTEGER'),
 (9, 'category', 'INTEGER'),
 (10, 'length', 'INTEGER'),
 (11, 'formula', 'STRING')

2. reps-25_batch_size-8_num_epochs-400.db: An sqlite3 database with two tables, 'status' and 'data'.
It contains the full learning data for the neural nets.
- 'status' same as for 1.
- 'data' contains the following columns:
 (0, 'category', 'INTEGER'),
 (1, 'rep', 'INTEGER'),
 (2, 'epoch', 'INTEGER'),
 (3, 'batch', 'INTEGER'),
 (4, 'loss', 'FLOAT')

3. complete_lengths.npy: A numpy array with shape (LoTs, categories) that gives the length of the minimal formula for each category in each LoT 
(including dual LoTs)

4. lengths_data.npy: Numpy array with shape (LoTs, categories) that has the lengths data for only one of each couple of dual LoTs.

5. learning_costs.pkl: Pandas df with columns (category, rep, effort) which contains the average loss (across batches and epochs) of each
repetition for each category.

6. freq_reg.pkl: List of two lists. 
- First list is linear regression for each LoT (learning efforts vs. minimal formulas). 
- Second list is AIC of the regression.

7. elbos.plk: Results of running variational inference on the linear model, which gives approximations of the bayesian evidence

8. spearman_ranks.npy: The Spearman rank correlations between the ANN learning efforts and minimal formula lengths for each LoT