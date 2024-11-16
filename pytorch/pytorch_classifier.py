import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class PyTorchClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
            self,
            hidden_layer_sizes = (128, 64, 32),
            activation: torch.nn.Module = torch.nn.ReLU(),
            dropout = 0.2,
            optimizer_config: dict = {"lr": 0.001, "weight_decay": 0.0001},
            epochs: int = 50,
            batch_size: int = 512,
            validation: bool = True,
            early_stopping: bool = False, 
            early_stop_thresh: int = 10,
            random_state: int = 100,
            verbose: int = True,
            device: str | torch.device = "cpu"
            ):
        """
        Neural network classifier implemented using PyTorch. Mostly sci-kit learn compatible.

        ## Parameters
            hidden_layer_sizes: array-like, default = (128, 64, 32)
                The number of neurons in each hidden layer.

            activation: torch.nn.Module or array-like, default = `torch.nn.ReLU()`
                Activation function to use for each hidden layer. Pass an array of activation functions to set it for each hidden layer.
                See: [https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
                
            dropout: float or array-like, default = 0.2
                Dropout probability for each hidden layer. Pass a float to set the probability for all hidden layers, 
                or pass an array of floats with the same length as `hidden_layer_sizes` to set the probability for each layer.
            
            optimizer_config: dict, default = `{"lr": 0.001, "weight_decay": 0.0001}`
                Pass a dictionary with keyword arguments to the Adam optimizer, 
                See: [https://pytorch.org/docs/stable/generated/torch.optim.Adam.html](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
            
            epochs: int, default = 50
                Maximum number of training iterations.
            
            batch_size: int, default = 512
                Number of samples to load per batch during training.

            validation: bool, default = True
                Whether to create a validation set from 10% of the training data in order to check for overfitting.
                The trained model will be the iteration with the highest score on the validation set.
            
            early_stopping: bool, default = False
                When using `validation`, whether to stop training after `early_stop_thresh` iterations without an improvement.

            early_stop_thresh: int, default = 10
                Number of iterations after which training will stop if there is no improvement in the validation set score.

            random_state: int, default = 100
                Seed for reproducibility. Note that the results may not be completely reproducible.
            
            verbose: bool, default = True
                Whether to print information about the loss for each epoch.
                
            device: `str` or `torch.device`, default = 'cpu'
                Specify which hardware device ('cpu', 'cuda', 'mps', etc.) to use for loading data, training, and inference.
                The correct version of PyTorch must be installed to use devices other than 'cpu'.
        """
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout = dropout
        self.optimizer_config = optimizer_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.early_stopping = early_stopping
        self.early_stop_thresh = early_stop_thresh
        self.random_state = random_state
        self.verbose = verbose
        self.device = device


    def _build_model(self) -> nn.Module:
        """
        Create a Sequential PyTorch model.
            
        ## Returns
            pytorch_model: torch.nn.Module  
        """
        if isinstance(self.dropout, float):
            dropout_probs = np.broadcast_to(self.dropout, np.shape(self.hidden_layer_sizes))
        else:
            dropout_probs = self.dropout
        if np.shape(dropout_probs) != np.shape(self.hidden_layer_sizes):
            raise ValueError("Dropout should be a number between 0 and 1, or an array of values the same length as hidden_layer_sizes")

        if isinstance(self.activation, nn.Module):
            activation_funcs = np.array([copy.deepcopy(self.activation) for _ in range(len(self.hidden_layer_sizes))])
        else:
            activation_funcs = self.activation
        if np.shape(activation_funcs) != np.shape(self.hidden_layer_sizes):
            raise ValueError("Activation should be an activation function, or an array of activation functions the same length as hidden_layer_sizes")

        pytorch_model = nn.Sequential(
            nn.Linear(self.n_features_, self.hidden_layer_sizes[0]),
            activation_funcs[0],
            nn.Dropout(dropout_probs[0])
        )

        for i in range(1, len(self.hidden_layer_sizes)):
            pytorch_model.append(nn.Linear(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i]))
            pytorch_model.append(activation_funcs[i])
            pytorch_model.append(nn.Dropout(dropout_probs[i]))
        
        pytorch_model.append(nn.Linear(self.hidden_layer_sizes[-1], self.n_classes_))

        return pytorch_model.to(torch.device(self.device))


    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Create the PyTorch optimizer.

        ## Returns
            optimizer: torch.optim.Optimizer
        """
        optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_config)
        return optimizer


    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


    def save_model(self, filename = "pytorch_model.pth"):
        """
        Save model weights to file. 
        Note: label names will need to be saved separately. These can be obtained with `np.unique(y_train)`.
        
        Alternatively, use the `save_checkpoint()` method to save everything.
        """
        check_is_fitted(self)
        torch.save(self.model.state_dict(), filename)
        print(f"Saved model weights to '{filename}'")


    def load_model(self, filename):
        """
        Load model weights from file.
        """
        self.set_seed(self.random_state)
        weights = torch.load(filename)
        try:
            self.n_features_ = weights[tuple(weights.keys())[0]].shape[1]
            self.n_classes_ = weights[tuple(weights.keys())[-1]].shape[0]
        except IndexError:
            raise Exception("Unable to get input/output dimensions from weights file.")
        
        if not self.__sklearn_is_fitted__():
            self.model = self._build_model()

        self.model.load_state_dict(weights)
        
        self._is_fitted = True
        return self


    def save_checkpoint(self, filename = "pytorch_checkpoint.tar"):
        """
        Save checkpoint (parameters, model state, optimizer state) to file. 
        Creates an uncompressed archive.
        """
        check_is_fitted(self)

        torch.save({
            'params': self.get_params(),

            'n_samples_': self.n_samples_,
            'n_features_': self.n_features_,
            'n_classes_': self.n_classes_,
            'classes_': self.classes_,
            'label_encoder_': self.label_encoder_,

            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'trained_epochs': self.trained_epochs
        }, filename)
        print(f"Saved checkpoint to '{filename}'")


    def load_checkpoint(self, filename):
        """
        Load checkpoint (parameters, model state, optimizer state) from file.
        """
        checkpoint = torch.load(filename)
        self.set_params(**checkpoint['params'])

        self.n_samples_ = checkpoint['n_samples_']
        self.n_features_ = checkpoint['n_features_']
        self.n_classes_ = checkpoint['n_classes_']
        self.classes_ = checkpoint['classes_']
        self.label_encoder_ = checkpoint['label_encoder_']

        self.set_seed(self.random_state)
        
        if not self.__sklearn_is_fitted__():
            self.model = self._build_model()
            self.optimizer = self._build_optimizer()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        self.trained_epochs = checkpoint['trained_epochs']
        
        self._is_fitted = True
        return self


    def change_device(self, device: str | torch.device):
        """
        Change model and data device
        """
        self.device = device
        if self.__sklearn_is_fitted__():
            self.model.to(torch.device(self.device))

        return self


    def set_seed(self, seed: int):
        """
        Change random_state

        ## Parameters
            seed: int
        """
        self.random_state = seed
        torch.manual_seed(self.random_state)
        self._torch_generator = torch.Generator().manual_seed(self.random_state)
        return self


    def get_tensor(self, arr, dtype: str | np.dtype = "float32") -> torch.Tensor:
        """
        Covert DataFrame or array to PyTorch tensor, then move it to model device.

        ## Parameters
            arr: array-like

            dtype: str or numpy.dtype
                Array will first be converted to a numpy.ndarray to allow conversion of DataFrames
        
        ## Returns
            tensor: torch.Tensor
        """
        tensor = torch.from_numpy(np.asarray(arr, dtype=dtype)).to(torch.device(self.device))
        return tensor


    def _train_loop(self, train_dataloader: DataLoader) -> float:
        """
        Backpropogation, and calculate training loss.

        ## Parameters
            train_dataloader: `torch.utils.data.DataLoader`
        
        ## Returns
            mean_loss: float
                Mean loss over all training samples
        """
        total_loss = 0
        for X, y in train_dataloader:
            pred = self.model(X)
            loss = self.loss_function(pred, y)
            total_loss += loss.item() * y.size(dim=0)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        mean_loss = total_loss / self.n_samples_
        return mean_loss


    def fit(self, X, y):
        """
        Fit the PyTorch model according to the training data.

        ## Parameters
            X: array like
            
            y: array with ndim == 1        
        """
        # Input validation
        self.set_seed(self.random_state)
        
        if int(self.epochs) > 0:
            self.epochs = int(self.epochs)
        else:
            raise ValueError("Epochs should be an positive integer.")
        
        if int(self.early_stop_thresh) > 0:
            self.early_stop_thresh = int(self.early_stop_thresh)
        else:
            raise ValueError("The early stopping threshold should be a positive integer")
        
        if int(self.batch_size) > 0:
            self.batch_size = int(self.batch_size)
        else:
            raise ValueError("Batch size should be a positive integer")

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]


        # Load training data
        if self.validation:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.10, random_state=self.random_state)
            X_val = self.get_tensor(X_val, dtype="float32")
            y_val = self.get_tensor(self.label_encoder_.transform(y_val), dtype="int64")

        train_data = list(
            zip(
                self.get_tensor(X, dtype="float32"), 
                self.get_tensor(self.label_encoder_.transform(y), dtype="int64")
            )
        )
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size, generator=self._torch_generator)
        
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.loss_function = torch.nn.CrossEntropyLoss()


        # Training loop
        best_loss = 1
        best_epoch = -1
        self.model.train()
        for epoch in range(self.epochs):
            self.loss = self._train_loop(train_dataloader)
            
            if self.validation:
                val_out = self.model(X_val)
                val_loss = self.loss_function(val_out, y_val).item()
                if self.verbose:
                    print(f"{epoch}\tVal loss: {val_loss:.3f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_weights = copy.deepcopy(self.model.state_dict())

                if self.early_stopping and epoch - best_epoch > self.early_stop_thresh:
                    self.model.load_state_dict(best_weights)
                    if self.verbose:
                        print(f"Early stopped training. Best epoch: {best_epoch}")
                    break
                
                if epoch + 1 == self.epochs:
                    self.model.load_state_dict(best_weights)
                    if self.verbose:
                        print(f"Finished training. Best epoch: {best_epoch}")
            
            else:
                if self.verbose:
                    print(f"{epoch}\tTrain loss: {self.loss:.3f}")
        
        self.trained_epochs = epoch
        self._is_fitted = True
        return self


    def train(self, X, y):
        """
        Alias for `fit` method.
        """
        return self.fit(X, y)


    def predict_proba(self, X) -> np.ndarray:
        """
        Get probability estimates.

        ## Parameters
            X: array-like of shape (n_samples, n_features)
                Data for which to get probabilities.

        ## Returns
            y_probs: ndarray of shape (n_samples, n_classes)
                Array of probabilities of labels for each sample.
                Label names are stored in `.classes_`
        """
        check_is_fitted(self)
        X = check_array(X)

        self.model.eval()
        with torch.no_grad():
            proba = self.model(self.get_tensor(X, dtype="float32")).softmax(dim=1).cpu().numpy()
        return proba


    def predict(self, X) -> np.ndarray:
        """
        Predict labels for samples.

        ## Parameters
            X: array-like of shape (n_samples, n_features)
                Data for which to get predictions.
        
        ## Returns
            y_pred: ndarray of shape (n_samples,)
                Vector of predictions for each sample in X.
        """
        proba = self.predict_proba(X)
        pred = self.classes_.take(np.argmax(proba, axis=1), axis=0)
        return pred


    def predict_proba1(self, X) -> np.ndarray:
        """
        For binary classification, get probability of label 1

        ## Parameters
            X: array-like of shape (n_samples, n_features)
                Data for which to get probabilities.

        ## Returns
            y_probs: ndarray of shape (n_samples,)
                Array of probabilities of label 1 for each sample.
        """
        proba = self.predict_proba(X)
        return proba[:,1]