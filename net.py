from ops import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = Encoder()
        self.cb = contextual_block()
        self.decoder = Decoder()

    def forward(self, x, m):

        h = torch.cat([x, m], 1)
        h = self.encoder(h)
        cs = self.cb(h, m)

        if self.training:
            inp = self.decoder(cs)
            cor = self.decoder(h)

            return inp, cor
        else:

            return self.decoder(cs)*(1-m) + x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = conv5x5(4, 64, stride=2)
        self.conv2 = conv5x5(64, 128, stride=2)
        self.conv3 = conv5x5(128, 256, stride=2)
        self.conv4 = conv5x5(256, 512, stride=2)
        self.conv5 = conv5x5(512, 512)

        self.final = ASPPModule(512)

    def forward(self, x, m):
        h = torch.cat([x, m], 1)

        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv3(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv4(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv5(h)

        return self.final(h)

class ASPPModule(nn.Module):
    def __init__(self, in_channels, rates=(1, 3, 6, 9)):
        super(ASPPModule, self).__init__()

        self.aspp = []
        for rate in rates:
            self.aspp.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                conv5x5(in_channels, in_channels, dilation=rate),
                nn.LeakyReLU(0.2),
                conv5x5(in_channels, 1),
            ))
        self.aspp = nn.ModuleList(self.aspp)

    def forward(self, x):

        out = []
        for conv in self.aspp:
            out.append(conv(x))

        return torch.cat(out, 1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layer1 = downsample(4, 32, 64)
        self.layer2 = downsample(64, 64, 128)
        self.layer3 = downsample(128, 128, 256)

        self.dcl1 = PAKA3x3(256, 256, dilation=2)
        self.dcl2 = PAKA3x3(256, 256, dilation=4)
        self.dcl3 = PAKA3x3(256, 256, dilation=8)
        self.dcl4 = PAKA3x3(256, 256, dilation=16)

    def forward(self, x):

        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        h = F.elu(self.dcl1(h))
        h = F.elu(self.dcl2(h))
        h = F.elu(self.dcl3(h))
        h = F.elu(self.dcl4(h))

        return h

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = upsample(256, 128)
        self.layer2 = upsample(128, 64)
        self.layer3 = upsample(64, 32)

        self.final = conv3x3(32, 3)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        h = self.final(h)
        h = torch.clamp(h, -1, 1)

        return h

class contextual_block(nn.Module):
    def __init__(self, scale=10):
        super(contextual_block, self).__init__()

        self.scale = scale

        self.softmax = nn.Softmax(dim=-1)

    def cb_single(self, x, m):
        channel, height, width = x.size()
        x_flat = x.view(channel, -1)
        m_flat = m.view(-1)
        idx = torch.nonzero(m_flat).squeeze(-1)

        bg = torch.index_select(x_flat, 1, idx)
        fg = x_flat

        CS = torch.matmul(fg.permute(1, 0), bg)
        CS = self.softmax(self.scale*CS)

        ACL = torch.matmul(bg, CS.permute(1, 0))
        out = ACL.view(channel, height, width)

        return out.unsqueeze(0)

    def forward(self, x, m):
        m = F.interpolate(m, size=x.shape[2:], mode='nearest')

        out = []
        for n in range(x.size()[0]):
            temp = self.cb_single(x[n], m[n])
            out.append(temp)

        return x*m + torch.cat(out, 0)*(1-m)
