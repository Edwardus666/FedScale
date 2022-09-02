# -*- coding: utf-8 -*-
import collections
import gc
import pickle
# The argparse module makes it easy to write user-friendly command-line interfaces.
# The argparse module also automatically generates help and usage messages # and issues errors when users give the program invalid arguments.
from argparse import Namespace

import torch
# This package contains the generated classes and enums for the Google Protobuf API for the job API.
import fedscale.core.channels.job_api_pb2 as job_api_pb2


from fedscale.core import commons
#  is used to manage client connections.
from fedscale.core.channels.channel_context import ClientConnections
# is used to submit jobs to the server.
from fedscale.core.execution.client import Client
from fedscale.core.execution.data_processor import collate, voice_collate_fn
#  is used to submit jobs to the server using reinforcement learning.
from fedscale.core.execution.rlclient import RLClient
from fedscale.core.logger.execution import *


class Executor(object):
    """Abstract class for FedScale executor.

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """
    def __init__(self, args):

        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu')
        self.num_executors = args.num_executors
        # ======== env information ========
        self.this_rank = args.this_rank  
        self.executor_id = str(self.this_rank)

        # ======== model and data ========
        self.model = self.training_sets = self.test_dataset = None
        self.temp_model_path = os.path.join(
            logDir, 'model_'+str(args.this_rank)+'.pth.tar')

        # ======== channels ========
        self.aggregator_communicator = ClientConnections(
            args.ps_ip, args.ps_port)

        # ======== runtime information ========
        self.collate_fn = None #The function used to collate data.
        self.task = args.task  #task name
        self.round = 0  # the round number 
        self.start_run_time = time.time()  #The time when the executor starts running.
        self.received_stop_request = False  # A flag indicating whether the executor has received a stop request.
        self.event_queue = collections.deque()

        super(Executor, self).__init__()

    def setup_env(self):
        """Set up experiments environment
        """
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")
        self.setup_seed(seed=1)

    def setup_communication(self):
        """Set up grpc connection
        """
        self.init_control_communication()
        self.init_data_communication()

    def setup_seed(self, seed=1):
        """Set random seed for reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        # https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed_all.html#torch-cuda-manual-seed-all
        torch.cuda.manual_seed_all(seed)
        
        
        np.random.seed(seed)
        random.seed(seed)
#         https://pytorch.org/docs/stable/notes/randomness.html 
#         results may not be reproducible between CPU and GPU executions, even when using identical seeds.
#         https://zhuanlan.zhihu.com/p/141063432 
        torch.backends.cudnn.deterministic = True

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
#         https://github.com/SymbioticLab/FedScale/blob/88ea51a967/fedscale/core/channels/channel_context.py#:~:text=def%20connect_to_server(self)%3A
        self.aggregator_communicator.connect_to_server()

    def init_data_communication(self):
        """In charge of jumbo data traffics (e.g., fetch training result)
        """
#         这里应该是当data traffic足够大时，需要自己去configure调用的
        pass

    def init_model(self):
        """Get the model architecture used in training

        Returns: 
            PyTorch or TensorFlow module: Based on the executor's machine learning framework, initialize and return the model for training
        
        """
#         这个function没看懂
        assert self.args.engine == commons.PYTORCH, "Please override this function to define non-PyTorch models"
        model = init_model()
        model = model.to(device=self.device)
        return model

    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        train_dataset, test_dataset = init_dataset()
#         https://github.com/SymbioticLab/FedScale/search?q=rl
        if self.task == "rl":
            return train_dataset, test_dataset
        # load data partitioner (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(
            data=train_dataset, args=self.args, numOfClass=self.args.num_class)
#         https://github.com/SymbioticLab/FedScale/search?q=partition_data_helper
        training_sets.partition_data_helper(
            num_clients=self.args.num_participants, data_map_file=self.args.data_map_file)

        testing_sets = DataPartitioner(
            data=test_dataset, args=self.args, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        logging.info("Data partitioner completes ...")

        if self.task == 'nlp':
#             https://github.com/SymbioticLab/FedScale/search?q=collate
            self.collate_fn = collate
        elif self.task == 'voice':
            self.collate_fn = voice_collate_fn

        return training_sets, testing_sets

    def run(self):
        """Start running the executor by setting up execution and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.model = self.init_model()
        self.training_sets, self.testing_sets = self.init_data()
        self.setup_communication()
        self.event_monitor()

    def dispatch_worker_events(self, request):
        """Add new events to worker queues
        
        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
        
        """
        self.event_queue.append(request)

    def deserialize_response(self, responses):
#         https://stackoverflow.com/questions/3316762/what-is-deserialize-and-serialize-in-json
        """Deserialize the response from server

        Args:
            responses (byte stream): Serialized response from server.

        Returns:
            ServerResponse defined at job_api.proto: The deserialized response object from server.
        
        """
#     https://docs.python.org/3/library/pickle.html 
# 用pickle -- Python 的object serialization来实现deserialize response from server
        return pickle.loads(responses)

    def serialize_response(self, responses):
#         client端在完成assigned job后， 序列化response，然后发给server端
        """Serialize the response to send to server upon assigned job completion

        Args:
            responses (string, bool, or bytes): Client responses after job completion.

        Returns:
            bytes stream: The serialized response object to server.
        
        """
#     用pickle的dumps()函数完成
        return pickle.dumps(responses)

    def UpdateModel(self, config):
        """Receive the broadcasted global model for current round

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model config
        
        """
#         Update the worker with the configuration of broadcasted global model 
        self.update_model_handler(model=config)

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:     
            tuple (int, dictionary): The client id and train result

        """
        client_id, train_config = config['client_id'], config['task_config']

#         init model
        model = None
#     正常处理 model 不为空的情况
        if 'model' in train_config and train_config['model'] is not None:
            model = train_config['model']

#             在下面的276行
        client_conf = self.override_conf(train_config)
        train_res = self.training_handler(
            clientId=client_id, conf=client_conf, model=model)

        # Report execution completion meta information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
#             https://github.com/SymbioticLab/FedScale/blob/3fa87b55579be6d2fbfd76e5e508527873551297/fedscale/core/channels/job_api_pb2.py
            job_api_pb2.CompleteRequest(
                client_id=str(client_id), executor_id=self.executor_id,
                event=commons.CLIENT_TRAIN, status=True, msg=None,
                meta_result=None, data_result=None
            )
        )
#         142行
#         https://github.com/SymbioticLab/FedScale/blob/88ea51a967bc277bdd64a1734cd595e1868063b2/fedscale/core/execution/executor.py#L79:~:text=def%20dispatch_worker_events(self%2C%20request)%3A
        self.dispatch_worker_events(response)

        return client_id, train_res

    def Test(self, config):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group
        
        Args:
            config (dictionary): The client testing config.
        
        """
#         338行
        test_res = self.testing_handler(args=self.args)
        test_res = {'executorId': self.this_rank, 'results': test_res}

        # Report execution completion information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=self.executor_id, executor_id=self.executor_id,
                event=commons.MODEL_TEST, status=True, msg=None,
                meta_result=None, data_result=self.serialize_response(test_res)
            )
        )
#         142行
        self.dispatch_worker_events(response)

    def Stop(self):
        """Stop the current executor
        """
#         https://github.com/SymbioticLab/FedScale/search?q=ClientConnections
        self.aggregator_communicator.close_sever_connection()
        self.received_stop_request = True

    def report_executor_info_handler(self):
        """Return the statistics of training dataset

        Returns:
            int: Return the statistics of training dataset, in simulation return the number of clients

        """
        return self.training_sets.getSize()

    def update_model_handler(self, model):
        """Update the model copy on this executor

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model

        """
        self.model = model
        self.round += 1

        # Dump latest model to disk
        with open(self.temp_model_path, 'wb') as model_out:
            pickle.dump(self.model, model_out)

    def load_global_model(self):
        """ Load last global model

        Returns:
            PyTorch or TensorFlow model: The lastest global model

        """
        with open(self.temp_model_path, 'rb') as model_in:
            model = pickle.load(model_in)
        return model

    def override_conf(self, config):
        """ Override the variable arguments for different client

        Args:
            config (dictionary): The client runtime config.

        Returns:
            dictionary: Variable arguments for client runtime config.

        """
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]
# https://stackoverflow.com/questions/11315010/what-do-and-before-a-variable-name-mean-in-a-function-signature
        return Namespace(**default_conf)

    def get_client_trainer(self, conf):
        """A abstract base class for client with training handler, developer can redefine to this function to customize the client training:

        Args:
            config (dictionary): The client runtime config.

        Returns:
            Client: A abstract base client class with runtime config conf.

        """
#         跳到from fedscale.core.execution.client import Client
        return Client(conf)

    def training_handler(self, clientId, conf, model=None):
        """Train model given client id
        
        Args:
            clientId (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result
        
        """
        # load last global model
        client_model = self.load_global_model() if model is None else model

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
#         https://github.com/SymbioticLab/FedScale/blob/3fa87b55579be6d2fbfd76e5e508527873551297/fedscale/core/execution/rlclient.py
# rl还是没明白是什么意思
        if self.args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets,
                                         batch_size=conf.batch_size, args=self.args,
                                         collate_fn=self.collate_fn
                                         )

            client = self.get_client_trainer(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)

#             training result
        return train_res

    def testing_handler(self, args):
        """Test model
        
        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            dictionary: The test result

        """
        evalStart = time.time()
        device = self.device
        model = self.load_global_model()
        if self.task == 'rl':
            client = RLClient(args)
            test_res = client.test(args, self.this_rank, model, device=device)
#             前三个变量不需要了，只要最后一个testResults
            _, _, _, testResults = test_res
        else:
            data_loader = select_dataset(self.this_rank, self.testing_sets,
                                         batch_size=args.test_bsz, args=args,
                                         isTest=True, collate_fn=self.collate_fn
                                         )

            if self.task == 'voice':
                criterion = CTCLoss(reduction='mean').to(device=device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(device=device)

            if self.args.engine == commons.PYTORCH:
                test_res = test_model(self.this_rank, model, data_loader,
                                      device=device, criterion=criterion, tokenizer=tokenizer)
            else:
                raise Exception(f"Need customized implementation for model testing in {self.args.engine} engine")

            test_loss, acc, acc_5, testResults = test_res
            logging.info("After aggregation round {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                         .format(self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))

#             python garbage collection
#             https://docs.python.org/3/library/gc.html#gc.collect
        gc.collect()

        return testResults

    def client_register(self):
        """Register the executor information to the aggregator
        """
        start_time = time.time()
        while time.time() - start_time < 180:
            try:
                response = self.aggregator_communicator.stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id=self.executor_id,
                        executor_id=self.executor_id,
                        executor_info=self.serialize_response(
                            self.report_executor_info_handler())
                    )
                )
                self.dispatch_worker_events(response)
                break
            except Exception as e:
                logging.warning(f"Failed to connect to aggregator {e}. Will retry in 5 sec.")
                time.sleep(5)

    def client_ping(self):
        """Ping the aggregator for new task
        """
#         aggregator_communicator.stub
#         https://github.com/SymbioticLab/FedScale/blob/3fa87b55579be6d2fbfd76e5e508527873551297/fedscale/core/channels/channel_context.py#:~:text=self.stub%20%3D%20job_api_pb2_grpc.JobServiceStub(self.channel)
# https://github.com/SymbioticLab/FedScale/blob/b31f072ae9cb2eeb63a49946bbf1946ffd942227/fedscale/core/channels/job_api_pb2.py       
            response = self.aggregator_communicator.stub.CLIENT_PING(job_api_pb2.PingRequest(
            client_id=self.executor_id,
            executor_id=self.executor_id
        ))
        self.dispatch_worker_events(response)

    def event_monitor(self):
        """Activate event handler once receiving new message
        """
        logging.info("Start monitoring events ...")
        self.client_register()

        while self.received_stop_request == False:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                if current_event == commons.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(client_id=str(client_id), executor_id=self.executor_id,
                                                    event=commons.UPLOAD_MODEL, status=True, msg=None,
                                                    meta_result=None, data_result=self.serialize_response(train_res)
                                                    ))
                    future_call.add_done_callback(lambda _response: self.dispatch_worker_events(_response.result()))

                elif current_event == commons.MODEL_TEST:
                    self.Test(self.deserialize_response(request.meta))

                elif current_event == commons.UPDATE_MODEL:
                    broadcast_config = self.deserialize_response(request.data)
                    self.UpdateModel(broadcast_config)

                elif current_event == commons.SHUT_DOWN:
                    self.Stop()

                elif current_event == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                self.client_ping()


if __name__ == "__main__":
    executor = Executor(args)
    executor.run()
