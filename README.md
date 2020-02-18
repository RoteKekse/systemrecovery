# Tensor Network Approaches for Learning Non-Linear Dynamical Laws - Code
This is the code concerning the above mentioned paper.
The project is structured as follows
- README.md - this file
- LICENSE   - license file
- scripts   - jupyter notebooks in ```python 3``` and ```R``` scripts to generate the plots

In scripts we have the file [```helpers.py```](https://github.com/anonymous-paper-2020/systemrecovery/blob/master/scripts/helpers.py) containing all sorts of helper functions concerning the performed experiments. This especially means that ```helpers.py``` include ```run_salsa()``` and ```run_als()``` which are the functions performing the SALSA and regualrized ALS simulations. Furthermore, we have the 4 jupyter notebooks:
- [```FPUT_salsa_model_a.ipynb```](https://github.com/anonymous-paper-2020/systemrecovery/blob/master/scripts/FPUT_salsa_model_a.ipynb) performing the SALSA simulation for Figure 5 in the paper
- [```FPTU_rals_model_b.ipynb```](https://github.com/anonymous-paper-2020/systemrecovery/blob/master/scripts/FPTU_rals_model_b.ipynb) performing the regularized ALS simulation for Figure 5 in the paper
- [```RANDOM_rals_model_b.ipynb```](https://github.com/anonymous-paper-2020/systemrecovery/blob/master/scripts/RANDOM_rals_model_b.ipynb) performing the regularized ALS simulation for Figure 6 in the paper
- [```RANDOM_FPTU_MEAN_rals_model_b.ipynb```](https://github.com/anonymous-paper-2020/systemrecovery/blob/master/scripts/RANDOM_FPTU_MEAN_rals_model_b.ipynb) performing the regularized ALS simulation for Figure 7 in the paper

All these notebooks are organized as follows:
- First we load different libraries.  With the exception of [xerus](https://libxerus.org) all libraries are standard. We use ```numpy```, ```xerus```, ```matplotlib``` and ```pandas```.
- Then we have a function for the exact solution. This is either the FPUT solution or the random variants of them in the respective format.
- Then we have a function to initialize the problem.
- Then we have the box containing  the configurations for the problem. We can specify for which dimensions and overservation sizes we want run the simulation. We can use the number of iterations and the number of runs for each pair of dimensions and observations. Furthermore, we can specify the name of the outputfile where we will store the results of the simulation (relative errors to the exact solution per sweep). This box looks more or less like this:
```python
# We choose different pairs of dimensions and samplesizes to run the algoirthm for.
data_noo_nos = [(6,1000),(6,1400),(6,1800),(6,2200),(6,2600),(6,3000),(6,3400),(6,3800),\
                (12,1400),(12,1900),(12,2400),(12,2900),(12,3400),(12,3900),(12,4400),(12,4900),\
               (18,1600),(18,2200),(18,2800),(18,3400),(18,4000),(18,4600),(18,5200),(18,5800)]
runs = 10 # each pair of dimension and samplesize is run 10 times
max_iter = 20 # we do 20 sweeps
output = 'data.csv'

tuples = []
for data in data_noo_nos:
    noo = data[0]
    nos = data[1] 
    for r in range(0,runs):
        tuples.append((noo,nos,r))
index = pd.MultiIndex.from_tuples(tuples, names=['d', 'm','runs'])           

# The results of each optimization is store in a Dataframe
df = pd.DataFrame(np.zeros([len(tuples),max_iter]), index=index) 
print(len(index))
print(data_noo_nos)
```
- Lastly, we have a loop calling the algorithms for the specific pairs of dimensions and number of observations run times.

Then we have the notebook [```AnalyseData.ipynb```](https://github.com/anonymous-paper-2020/systemrecovery/blob/master/scripts/AnalyseData.ipynb) which analyses the results of the four simulation types above and creates tables which can directly be used for the R plots Figure 5,6,7 and the Table 1 in the paper. Finally, the R plots for the three plots in the paper are stored in [```scripts/plots```](https://github.com/anonymous-paper-2020/systemrecovery/tree/master/scripts/plots).


## Simulation Environment
Furthermore, we have a simulation environment where the listed notebooks should run out of the box. For this visit [SimEnv](https://anonymous.rotekekse.de), a password should have been given to you. **Please keep in mind** the computational power is limited (4x2.5Ghz) and slower than the computer used in the experiments in the paper (4x3.5Ghz). So use it only for small tests.
