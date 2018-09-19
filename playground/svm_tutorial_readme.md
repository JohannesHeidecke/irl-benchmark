The SVM tutorial requires a changed version of the openAI baselines.

To use it, first follow the installation instructions here:

https://github.com/openai/baselines

(up to `pip install -e .`)



Then: go to the new baselines directory and replace the files

`/run.py`

`/trpo_mpi/trpo_mpi.py` 

with the files of the same name in `playground/baselines_changes`



Now you should be able to run the SVM-tutorial notebook.