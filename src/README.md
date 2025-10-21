


## Execution

### Native

Install the required python3 libraries:
```bash
pip install -r requirements
```
Require cython compilation of the C/C++ libraries
```bash
python3 setup.py build_ext --inplace
```

In the next step FLEx required some local dataset to be downloaded and stored. For this purpose execute:
```bash
python3 unified-data-splitter.py
```
This script will create dataset folder for multiple pre-defined datasets [breast_cancer, income, mushroom_splits]. In the next steps while we will execute the clients we should select one of the datasets from: '/src/datasets' folder.

Now we are ready to execute FLEx server and clients with following commands (in case of local execution please run each client/server in different terminal instance):
```bash
#this script will start server instance over the localhost ip address and listen in port 8080 for client connection
#also server will wait for 2 clients to participate in the federated training
# with weighted feature aggregation methodology and 40% degree of freedom
python3 server.py --port 8080 --clients 2 --fe_weighted --freedom 0.4
```

```bash
#this script will start single client instance which will connect to the server in address 127.0.0.1:8080 
#in this case the client will use income dataset with Gaussian Naive Bayes model as classifier
# --- Lasso feature selection method
# --- with 40% degree of freedom for the features  
python client.py  --ip 127.0.0.1 --host 127.0.0.1 --port 8080 --dataset income --model gnb --method lasso --freedom 0.4
```
Alternatively:
Run runner.py ...


### Docker
1.Run the Docker deamon
2.Run with custom configuration (scale and number of clients should be the same):
```bash
EXPERIMENT_TIME=$(date +%Y%m%d_%H%M%S) \
NUM_CLIENTS=5 \
MODEL=SGDC \
FS_METHOD=lasso \
FREEDOM_RATE=0.7 \
DATASET=mushroom \
docker-compose up --scale client=5
```

