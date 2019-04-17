### Weight Lifting Exercises Classification with Azure Databricks and Azure Machine Learning service

#### Instructions to run this tutorial. Here we also include links for more information about the key concepts behind Azure Databricks and Azure ML service:

1. Follow the instructions [here](https://docs.microsoft.com/en-us/azure/azure-databricks/quickstart-create-databricks-workspace-portal) until the step named "Create a Spark cluster in Databricks" to create an Azure Databricks Workspace and a compute cluster.

2. Follow the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#install-the-correct-sdk-into-a-databricks-library) to install the Azure ML SDK into the Databricks cluster. The base SDK **azureml-sdk[databricks]** is sufficient for this tutorial.

3. Download the notebooks and follow the instructions [here](https://docs.azuredatabricks.net/user-guide/notebooks/notebook-manage.html#import-a-notebook) to import them to your workspace.

4. Open the **WLE Classification (Python).ipynb** notebook and click on "Run All" at the top of the notebook to run it in your own environment. This is a standalone notebook (not dependent on the Azure ML service SDK). It will train and save a simple classification model to be further operationalized using Azure ML service model deploying capabilities. Please see [here](https://docs.azuredatabricks.net/user-guide/notebooks/notebook-use.html) if you want to learn more about working with Databricks Notebooks.

5. Open the **AML Operationalization.ipynb** notebook and click on "Run All" at the top of the notebook to run it in your own environment. This will create an [Image](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-azure-machine-learning-architecture#image) with a score script that uses your trained model for inference and a [Deployment](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-azure-machine-learning-architecture#deployment) environment from where your score script will be exposed as a web service.

6. Open the **AML Model Consumption.ipynb** notebook and click on "Run All" at the top of the notebook to run it in your own environment. This will construct an HTTP request to the deployed model, passing an instance of the input data to be classified. Please see [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-consume-web-service) for more information about Azure ML web services consumption.

#### To learn more:

[Azure ML service notebooks on GitHub](https://github.com/Azure/MachineLearningNotebooks)

[Microsoft Learn AI - Anomaly Detection & Predictive Maintenance](https://github.com/Azure/LearnAI-ADPM)

[Microsoft Learn AI - Custom AI Airlift](https://github.com/Azure/LearnAI-CustomAI-Airlift)
