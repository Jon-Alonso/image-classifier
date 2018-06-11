import torch
from torch.autograd import Variable


class Trainer(object):
    def __init__(self, scheduler, optimizer, criterion):

        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion
        self._gpu = False
        self._debug = False

    @property
    def gpu(self):
        """
        Use the GPU instead of the CPU.
        """
        return self._gpu

    @gpu.setter
    def gpu(self, value):
        self._gpu = value

    @property
    def debug(self):
        """
        Prints stats if set to True.
        """
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value

    def train(self, model, dataloader, dataset_size):
        """
        Train the model using the specified dataloader,
        :param model: model
        :param dataloader: DataLoader
        :param dataset_size: The number of images in the dataset
        """
        self.scheduler.step()
        model.train(True)

        loss, accuracy = self.learn(model, dataloader)

        if self._debug:
            print('Train Loss: {:.4f} Accuracy: {:.4f}'.format(
                loss / dataset_size,
                accuracy / dataset_size
            ))


    def validate(self, model, dataloader, dataset_size):
        """
        Validate the model using the specified dataloader,
        :param model: model
        :param dataloader: DataLoader
        :param dataset_size: The number of images in the dataset
        """
        model.train(False)

        loss, accuracy = self.learn(model, dataloader)

        if self._debug:
            print('Validation Loss: {:.4f} Accuracy: {:.4f}'.format(
                loss / dataset_size,
                accuracy / dataset_size
            ))


    def learn(self, model, dataloader):
        """
        Feed-forward all the data in the dataloader.
        :param model: model
        :param dataloader: Dataloader
        :returns: loss, accuracy
        """
        loss = 0.0
        accuracy = 0

        for data in dataloader:
            loss, accuracy = self.evaluate_data(model, data)
            loss += loss
            accuracy += accuracy

        return loss, accuracy

    def evaluate_data(self, model, data, persist=False):
        """
        Feed-forward the data into the model.
        :param model: model
        :param data: np.array
        :param persist: boolean True if model should use back propagation
        :returns: loss, accuracy
        """
        inputs, labels = self.create_vars(data)

        self.optimizer.zero_grad()
        outputs = model(inputs)
        _, predictions = torch.max(outputs.data, 1)
        loss = self.criterion(outputs, labels)

        if persist:
            loss.backward()
            self.optimizer.step()

        return self.calc_loss(loss, inputs), self.calc_accuracy(labels, predictions)

    def create_vars(self, data):
        """
        Generate the input and label variables.
        :param data: np.array
        :returns: Variable, Variable
        """
        inputs, labels = data

        if self._gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        return inputs, labels

    def calculate_loss(self, loss, inputs):
        """
        Calculate the total loss.
        :param loss: np.array
        :param inputs: np.array
        :returns: int
        """
        return loss.data[0] * inputs.size(0)

    def calc_accuracy(self, labels, predictions):
        """
        Calculate the predictions accuracy.
        :param labels: np.array
        :param predictions: np.array
        :returns: np.array
        """
        return torch.sum(predictions == labels.data)