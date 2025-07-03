
### Prerequisites:
Ensure you have Python version: Python 3.11.

### Install Dependencies:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run simulation:

To run experiments:
```sh
# Loop over strategy configurations in experiment folder, in model/conf/
python main.py --name experiment_name

```

### Visualize training results:
Outputs in results/plots
```sh
cd results

# Selection counts plot (Fig. 3)
python plot_selection.py noiseexp --selection

# Convergence and Emissions plot (Fig. 4 and 5) 
python plot_exp1.py noiseexp --split

# Selection plots with intensities (Fig. 6)
python plot_selection.py budget

# Carbon Budgetting (Fig. 7) 
python plot_budgets.py budget noisybudget 

# Dditional eval (Fig. 8) 
python plot_budget_accuracy.py tinyimagenetbudget/ noisetinyim/ cifar100budget/ noisecifar100/ --output_name final_plot --folder_names "Clean Data" "Data Noise Added" "Clean Data" "Data Noise Added"

# Unused plot - Convergence and Selectiont count with average C02 intensities
python plot_combined_new.py cifar100budget/ noisecifar100/  tinyimagenetbudget/ noisetinyim/  --second_dataset --output_name combined_visualization
```    

