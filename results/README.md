The countless runs required to test the different configurations + schedulers can be found in this folder. Each results folder has a uuid and contains the following files:

1. `args.json` - The exact arguments used to run the tuning script for later analysis
2. `ax_client.json` - The entire Ax client object, serialized to JSON, can later be loaded via `ax.load(ax_client_json)` -- later used to compute hypervolume, analyse the Pareto frontier, etc.
3. `pareto.html` - A navigable Plotly chart of the Pareto frontier.
4. `pareto.png` - A static image of the Pareto frontier.
5. `tuning_results_df.csv` - The final results of the hyperparameter tuning, with each trial's final metrics.
6. `tuning_results.pkl` - The entire `tune` object serialized to a pickle file. 