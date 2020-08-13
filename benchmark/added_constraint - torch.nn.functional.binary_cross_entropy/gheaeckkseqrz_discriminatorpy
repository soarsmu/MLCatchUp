import torch
import encoder

class Discriminator(torch.nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        self.e = encoder.Encoder(nc)
        self.fc = torch.nn.Linear(nc*4, 1)
        self.nc = nc

    def next_step(self):
        self.e.next_step()

    def forward(self, x):
        # x = torch.nn.functional.dropout(x)
        x = self.e(x)
        assert x.shape[2] == 2 and x.shape[3] == 2
        x = x.view(-1, self.nc*4)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def train(self, positive, negative):
        self.zero_grad()
        prediction_positive = self(positive)
        self.e.alpha = min(1, self.e.alpha - (1/(2048*8)))
        ground_truth_positive = torch.ones(prediction_positive.shape).cuda()
        loss_positive = torch.nn.functional.binary_cross_entropy(prediction_positive, ground_truth_positive)

        prediction_negative = self(negative)
        self.e.alpha = min(1, self.e.alpha - (1/(2048*8)))
        ground_truth_negative = torch.zeros(prediction_negative.shape).cuda()
        loss_negative = torch.nn.functional.binary_cross_entropy(prediction_negative, ground_truth_negative)
        
        loss = loss_positive + loss_negative;
        if torch.sum(prediction_positive) < positive.shape[0] * 0.8 or torch.sum(prediction_negative) > negative.shape[0] * 0.2:
            loss.backward()
        return prediction_positive, prediction_negative

    def teach(self, generated_samples):
        prediction_generated = self(generated_samples)
        target_generated = torch.ones(prediction_generated.shape).cuda()
        return torch.nn.functional.binary_cross_entropy(prediction_generated, target_generated), prediction_generated

if __name__ == "__main__":
    d = Discriminator(4)
    t = torch.Tensor(1, 3, 4, 4)
    for i in range(6):
        print("Step", i)
        print("Input", t.shape)
        output = d(t)
        t = torch.nn.functional.interpolate(t, scale_factor=2)
        d.next_step()
        print("Output", output.shape)
