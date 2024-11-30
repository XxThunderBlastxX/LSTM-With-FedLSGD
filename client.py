import socket
import pickle
import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

class FederatedLearningClient:
    def __init__(self, server_host='localhost', server_port=65432):
        # Load and preprocess MNIST data
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data()

        # Normalize data
        self.train_X = self.train_X / 255.0
        self.test_X = self.test_X / 255.0

        # Convert labels to one-hot encoding
        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)

        # Server connection details
        self.server_host = server_host
        self.server_port = server_port

        # Initialize model
        self.model = self.create_model()

    def create_model(self):
        # Create LSTM model for MNIST classification
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

    def connect_to_server(self):
        # Create socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.server_host, self.server_port))
        return client_socket

    def receive_initial_weights(self, client_socket):
        try:
            # Receive initial global model weights
            global_model_data = self.recv_msg(client_socket)
            if global_model_data:
                global_model_weights = pickle.loads(global_model_data)['weights']

                # Update local model with global weights
                self.model.set_weights(global_model_weights)
                print("Initialized local model with global weights")
                return True
            return False
        except Exception as e:
            print(f"Error receiving initial weights: {e}")
            return False

    def train_local_model(self, epochs=5, batch_size=32):
        # Local model training
        history = self.model.fit(
            self.train_X, self.train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history

    def send_updated_weights(self, client_socket):
        try:
            # Prepare model data to send
            model_data = {
                'weights': self.model.get_weights(),
                'dataset_size': len(self.train_X)
            }

            # Send updated model weights to server
            self.send_msg(client_socket, model_data)
            print("Sent updated model weights to server")
            return True
        except Exception as e:
            print(f"Error sending updated weights: {e}")
            return False

    def evaluate_model(self):
        # Evaluate model performance
        loss, accuracy = self.model.evaluate(self.test_X, self.test_y)
        print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

def main():
    # Create federated learning client
    client = FederatedLearningClient()

    # Connect to server
    print("Connecting to server...")
    client_socket = client.connect_to_server()

    try:
        # Receive initial global model weights
        if client.receive_initial_weights(client_socket):
            # Local training
            print("Starting local model training...")
            client.train_local_model()

            # Send updated model weights to server
            print("Sending updated model weights...")
            client.send_updated_weights(client_socket)

            # Evaluate model
            print("Evaluating model performance...")
            client.evaluate_model()
        else:
            print("Failed to receive initial weights")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close socket connection
        client_socket.close()

if __name__ == "__main__":
    main()
