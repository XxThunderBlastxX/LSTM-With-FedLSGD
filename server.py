import socket
import pickle
import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

class FederatedLearningServer:
    def __init__(self, host='localhost', port=65432, max_clients=5):
        # Initialize server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(max_clients)

        # Initialize global model
        self.global_model = self.create_base_model()

        # Aggregation parameters
        self.client_weights = []
        self.client_sizes = []

    def create_base_model(self):
        # Create the same model architecture as in the client
        model = Sequential([
            LSTM(50, input_shape=(28, 28)),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def send_msg(self, sock, msg):
        # Serialize the message
        msg = pickle.dumps(msg)
        # Send message length first
        sock.sendall(struct.pack('>I', len(msg)))
        # Send the message
        sock.sendall(msg)

    def recv_msg(self, sock):
        # Receive message length
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        # Unpack message length
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Receive entire message
        return self.recvall(sock, msglen)

    def recvall(self, sock, n):
        # Helper function to receive n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def weighted_average_model(self):
        # Aggregate model weights from clients
        if not self.client_weights:
            return

        new_weights = []
        for layer_weights in zip(*self.client_weights):
            # Compute weighted average of weights
            layer_aggregated = np.average(layer_weights, axis=0, weights=self.client_sizes)
            new_weights.append(layer_aggregated)

        # Set the global model weights
        self.global_model.set_weights(new_weights)

        # Reset client weights and sizes
        self.client_weights = []
        self.client_sizes = []

    def start_server(self):
        print("Federated Learning Server Started...")

        while True:
            # Accept client connection
            client_socket, address = self.server_socket.accept()
            print(f"Connection from {address}")

            try:
                # Send initial global model weights to the client
                initial_weights = self.global_model.get_weights()
                self.send_msg(client_socket, {
                    'weights': initial_weights
                })

                # Receive client model weights
                data = self.recv_msg(client_socket)
                if not data:
                    continue

                client_model_data = pickle.loads(data)

                # Store client weights and dataset size
                self.client_weights.append(client_model_data['weights'])
                self.client_sizes.append(client_model_data['dataset_size'])

                # Aggregate weights
                self.weighted_average_model()
                print("Global model updated with aggregated weights")

            except Exception as e:
                print(f"Error processing client: {e}")

            finally:
                client_socket.close()

if __name__ == "__main__":
    server = FederatedLearningServer()
    server.start_server()
