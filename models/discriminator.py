import torch
import torch.nn as nn


class LinearDiscriminator(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 input_dims: int = 256,
                 hidden_dims: int = 128) -> None:
        """
            num_classes: number of classes for classification
            hidden_dims: hidden dims for the model
        """
        super().__init__()
        self.predictor = nn.Sequential(nn.Linear(input_dims, hidden_dims),
                                       nn.LeakyReLU(),
                                       nn.Linear(hidden_dims, num_classes))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self._initialize_weights()

    def forward(self, latent_f, pred_f, use_sigmoid: bool=True, use_softmax: bool=False):
        pred = self.predictor(latent_f)

        assert not (use_sigmoid and use_softmax), "Cannot use both sigmoid and softmax"
        if use_sigmoid:
            pred = self.sigmoid(pred)
        elif use_softmax:
            pred = self.softmax(pred)
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


class ConditionLinearDiscriminator(LinearDiscriminator):
    def __init__(self,
                 num_classes: int,
                 num_pred_classes: int = 1,
                 input_dims: int = 256,
                 hidden_dims: int = 128) -> None:
        super().__init__(num_classes, input_dims*2, hidden_dims)
        self.pred_embedder = nn.Linear(num_pred_classes, input_dims)
        self.softmax = nn.Softmax(dim=1)
        self._initialize_weights()
        
    def forward(self, latent_f, pred_f, use_sigmoid: bool=False, use_softmax: bool=False):
        pred_embed = self.pred_embedder(pred_f)
        joint_embed = torch.concat((latent_f, pred_embed), dim=1)
        pred = self.predictor(joint_embed)

        assert not (use_sigmoid and use_softmax), "Cannot use both sigmoid and softmax"
        if use_sigmoid:
            pred = self.sigmoid(pred)
        elif use_softmax:
            pred = self.softmax(pred)
        return pred


def demo_linear_discriminator():
    latent_f = torch.randn((64, 256))
    pred_f = torch.randn((64, 1))

    model = LinearDiscriminator()
    pred = model(latent_f, pred_f)

    print(pred.shape)


def demo_condition_discriminator():
    latent_f = torch.randn((64, 256))
    pred_f = torch.randn((64, 1))

    model = ConditionLinearDiscriminator(num_classes=2)
    pred = model(latent_f, pred_f)

    print(pred.shape)


if __name__ == "__main__":
    demo_linear_discriminator()
    demo_condition_discriminator()

