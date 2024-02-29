import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import stable_exp, negative_binom_loss


SUB_1 = 64
SUB_2 = 32


class Exp(nn.Module):
    """Applies torch.exp, clamped to improve stability during training"""

    def __init__(self, minimum=1e-5, maximum=1e6):
        """Values taken from DCA"""
        super(Exp, self).__init__()

    def forward(self, input):
        exp_val = stable_exp(input)
        return exp_val


class NegativeBinomialLoss(nn.Module):
    """
    Negative binomial loss. Preds should be a tuple of (mean, dispersion)
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        eps: float = 1e-10,
        l1_lambda: float = 0.0,
        mean: bool = True,
    ):
        super(NegativeBinomialLoss, self).__init__()
        self.loss = negative_binom_loss(
            scale_factor=scale_factor,
            eps=eps,
            mean=mean,
            debug=True,
        )
        self.l1_lambda = l1_lambda

    def forward(self, preds, theta, encoded, target):
        loss = self.loss(
            preds=preds,
            theta=theta,
            truth=target,
        )
        loss += self.l1_lambda * torch.abs(encoded).sum()
        return loss


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.w_3 = nn.Conv1d(d_in, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm([d_in, 1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.unsqueeze(-1)
        output = self.w_2(F.relu(self.w_1(output)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual.unsqueeze(-1))
        output = self.w_3(output)
        output = output.squeeze(-1)
        return output


class PositionwiseResidualFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid, bias=False)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_hid, bias=False)  # position-wise
        self.w_3 = nn.Linear(d_hid, d_in, bias=False)  # position-wise
        self.batch_norm = nn.BatchNorm1d(d_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.w_2(F.gelu(self.w_1(x)))
        output = self.dropout(self.batch_norm(output))
        output = self.w_3(output)
        output = output + residual
        return output


class PositionwisePairFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_a = nn.Linear(d_in, d_hid)  # position-wise
        self.w_g = nn.Linear(d_in, d_hid)  # position-wise

        self.w_d = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in * 3, d_in),
            nn.PReLU(),
            nn.Linear(d_in, d_in // 2),
            nn.PReLU(),
            nn.Linear(d_in // 2, 1)

        )

    def forward(self, anchor, guess):
        ac = self.w_a(anchor)
        ge = self.w_g(guess)
        ge2ac_score = F.softmax(ge @ ac.transpose(0, 1), dim=-1)
        ge2ac = ge2ac_score @ anchor

        dis_x = torch.cat((guess, ge2ac, guess - ge2ac), dim=-1)
        output = self.w_d(dis_x)
        return output


class Encoder(nn.Module):
    ''' Addapted from https://github.com/wukevin/babel/blob/main/babel/models/autoencoders.py
    '''

    def __init__(self, input_dim, z_dim, activation=nn.PReLU):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

        self.encode1 = nn.Linear(self.input_dim, SUB_1)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(SUB_1)
        self.act1 = activation()

        self.encode2 = nn.Linear(SUB_1, SUB_2)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(SUB_2)
        self.act2 = activation()

        self.encode3 = nn.Linear(SUB_2, self.z_dim)
        nn.init.xavier_uniform_(self.encode3.weight)

    def forward(self, x):
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        x = self.encode3(x)
        return x


class Decoder(nn.Module):
    ''' Addapted from https://github.com/wukevin/babel/blob/main/babel/models/autoencoders.py
    '''

    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        activation=nn.PReLU,
        final_activation=[Exp(), nn.Softplus(), nn.Sigmoid()]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        assert len(final_activation) == 3
        self.final_activation = final_activation

        self.decode1 = nn.Linear(self.z_dim, SUB_2)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(SUB_2)
        self.act1 = activation()

        self.decode2 = nn.Linear(SUB_2, SUB_1)
        nn.init.xavier_uniform_(self.decode2.weight)
        self.bn2 = nn.BatchNorm1d(SUB_1)
        self.act2 = activation()

        self.decode31 = nn.Linear(SUB_1, self.input_dim)
        nn.init.xavier_uniform_(self.decode31.weight)
        self.decode32 = nn.Linear(SUB_1, self.input_dim)
        nn.init.xavier_uniform_(self.decode32.weight)
        self.decode33 = nn.Linear(SUB_1, self.input_dim)
        nn.init.xavier_uniform_(self.decode33.weight)

    def forward(self, x, size_factors=None):
        x = self.act1(self.bn1(self.decode1(x)))
        x = self.act2(self.bn2(self.decode2(x)))
        retval1 = self.final_activation[0](self.decode31(x))
        retval2 = self.final_activation[1](self.decode32(x))
        retval3 = self.final_activation[2](self.decode33(x))

        return retval1, retval2, retval3


class TestEnc(nn.Module):
    def __init__(self, input_dim):
        super(TestEnc, self).__init__()
        self.input_dim = input_dim
        self.enc1 = nn.Linear(input_dim, 640)
        self.bn1 = nn.BatchNorm1d(640)
        self.act1 = nn.PReLU()
        self.enc2 = nn.Linear(640, 320)
        self.bn2 = nn.BatchNorm1d(320)
        self.act2 = nn.PReLU()
        self.enc3 = nn.Linear(320, 20)
        self.enc4 = nn.Linear(320, 20)

    def forward(self, X):
        X = self.enc1(X)
        X = self.bn1(X)
        X = self.act1(X)
        X = self.enc2(X)
        X = self.bn2(X)
        X = self.act2(X)
        mean = self.enc3(X)
        logvar = self.enc4(X)

        return mean, logvar


class TestDec(nn.Module):
    def __init__(self, input_dim):
        super(TestDec, self).__init__()
        self.input_dim = input_dim
        self.dec1 = nn.Linear(20, 320)
        self.bn1 = nn.BatchNorm1d(320)
        self.act1 = nn.PReLU()
        self.dec2 = nn.Linear(320, 640)
        self.bn2 = nn.BatchNorm1d(640)
        self.act2 = nn.PReLU()
        self.dec31 = nn.Linear(640, self.input_dim)

    def forward(self, X):
        X = self.dec1(X)
        # X = self.bn1(X)
        X = self.act1(X)
        X = self.dec2(X)
        # X = self.bn2(X)
        X = self.act2(X)

        mu, theta, pi = self.dec31(X), None, None
        return mu, theta, pi


class SplitEnc(nn.Module):
    ''' Addapted from https://github.com/wukevin/babel/blob/main/babel/models/autoencoders.py
    '''

    def __init__(self, input_dim, z_dim):
        super(SplitEnc, self).__init__()
        self.input_dim = input_dim
        self.split_layer = nn.ModuleList()
        for n in self.input_dim:
            # assert isinstance(n, int)
            layer1 = nn.Linear(n, SUB_1)
            nn.init.xavier_uniform_(layer1.weight)
            bn1 = nn.BatchNorm1d(SUB_1)
            act1 = nn.PReLU()
            layer2 = nn.Linear(SUB_1, SUB_2)
            nn.init.xavier_uniform_(layer2.weight)
            bn2 = nn.BatchNorm1d(SUB_2)
            act2 = nn.PReLU()
            self.split_layer.append(
                nn.ModuleList([layer1, bn1, act1, layer2, bn2, act2])
            )

        self.enc2 = nn.Linear(SUB_2*len(input_dim), z_dim)
        self.bn2 = nn.BatchNorm1d(z_dim)
        self.mean_enc = nn.Linear(z_dim, z_dim)
        self.var_enc = nn.Linear(z_dim, z_dim)

        nn.init.xavier_uniform_(self.enc2.weight)
        nn.init.xavier_uniform_(self.mean_enc.weight)
        nn.init.xavier_uniform_(self.var_enc.weight)

    def forward(self, x):
        xs = torch.split(x, self.input_dim, dim=1)
        #print(x)
        #print(x.type())
        #print(xs)
        #print(xs[0].type())
        assert len(xs) == len(self.input_dim)
        enc_chroms = []
        for init_mod, chrom_input in zip(self.split_layer, xs):
            for f in init_mod:
                chrom_input = f(chrom_input)
            enc_chroms.append(chrom_input)
        enc1 = torch.cat(enc_chroms, dim=1)

        chromosome_enc = self.bn2(self.enc2(enc1))
        mean = self.mean_enc(chromosome_enc)
        log_var = self.var_enc(chromosome_enc)
        return mean, log_var


class SplitDec(nn.Module):
    ''' Addapted from https://github.com/wukevin/babel/blob/main/babel/models/autoencoders.py
    '''

    def __init__(self, input_dim, z_dim):
        super(SplitDec, self).__init__()
        self.input_dim = input_dim
        self.dec1 = nn.Linear(z_dim, len(self.input_dim) * SUB_2)
        self.bn1 = nn.BatchNorm1d(len(self.input_dim) * SUB_2)
        self.act1 = nn.PReLU()
        self.split_layer = nn.ModuleList()
        for n in self.input_dim:
            # assert isinstance(n, int)
            layer1 = nn.Linear(SUB_2, SUB_1)
            nn.init.xavier_uniform_(layer1.weight)
            bn1 = nn.BatchNorm1d(SUB_1)
            act1 = nn.PReLU()
            layer2 = nn.Linear(SUB_1, n)
            nn.init.xavier_uniform_(layer2.weight)
            self.split_layer.append(
                nn.ModuleList([layer1, bn1, act1, layer2])
            )

    def forward(self, x):
        x = self.act1(self.bn1(self.dec1(x)))
        xs = torch.chunk(x, chunks=len(self.input_dim), dim=1)
        rec_chroms = []
        for processors, chrom_input in zip(self.split_layer, xs):
            for f in processors:
                chrom_input = f(chrom_input)
            rec_chroms.append(chrom_input)
        rec = torch.cat(rec_chroms, dim=1)
        return rec


class NaiveAffineTransform(nn.Module):
    def __init__(self, input_dim, z_dim, affine_num, reverse=False) -> None:
        super().__init__()
        self.input_dim = input_dim

        # affine matrix init with identity affine
        direction = -1 if reverse else 1
        self.affine_matrix = nn.Parameter(torch.stack([torch.randn(input_dim, input_dim).flatten() * direction for _ in range(affine_num)]))
        self.affine_offset = nn.Parameter(torch.randn(affine_num, input_dim))

        # regressor for the affine transform selection
        self.fc_loc = nn.Sequential(
            nn.Linear(input_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, affine_num)
        )

    def forward(self, x):
        soft_idx = F.softmax(self.fc_loc(x), dim=-1)
        affine_matrix = torch.mm(soft_idx, self.affine_matrix)
        affine_matrix = affine_matrix.view(-1, self.input_dim, self.input_dim)  # [b, d, d]
        affine_offset = torch.mm(soft_idx, self.affine_offset)
        affine_offset = affine_offset.unsqueeze(-1)  # [b, d, 1]

        output = x.unsqueeze(-1)
        output = torch.bmm(affine_matrix, output) + affine_offset
        output = output.squeeze(-1)
        return output


class AffineTransform(nn.Module):
    def __init__(self, input_dim, z_dim, affine_num, affine_layer_num=3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.affine_layer_num = affine_layer_num

        # affine matrix init
        self.affine_matrices = nn.ParameterList(
            [nn.Parameter(torch.stack([torch.randn(input_dim, input_dim).flatten() for _ in range(affine_num)])) for _ in range(affine_layer_num)]
        )
        self.affine_offsets = nn.ParameterList(
            [nn.Parameter(torch.randn(affine_num, input_dim)) for _ in range(affine_layer_num)]
        )

        # regressor for the affine transform selection
        self.fc_loc = nn.Sequential(
            nn.Linear(input_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, affine_num)
        )
        self.act = nn.PReLU()

    def forward(self, x):
        soft_idx = F.softmax(self.fc_loc(x), dim=-1)

        output = x.unsqueeze(-1)
        for i in range(self.affine_layer_num):
            affine_matrix = torch.mm(soft_idx, self.affine_matrices[i])
            affine_matrix = affine_matrix.view(-1, self.input_dim, self.input_dim)  # [b, d, d]
            affine_offset = torch.mm(soft_idx, self.affine_offsets[i])
            affine_offset = affine_offset.unsqueeze(-1)  # [b, d, 1]

            # do affine transform
            output = torch.bmm(affine_matrix, output) + affine_offset

            # do activation until output
            if i < self.affine_layer_num - 1:
                output = self.act(output)

        output = output.squeeze(-1)
        return output


class NaiveAffineDiscriminator(nn.Module):
    def __init__(self, input_dim, affine_num, dropout=0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.affine_num = affine_num

        self.w_d = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.PReLU(),
            nn.Linear(self.input_dim // 2, self.affine_num)
        )

    def forward(self, x, affine_index=None):
        score = self.w_d(x)

        if affine_index is None:
            return score
        else:
            if isinstance(affine_index, torch.Tensor):
                index = affine_index
            else:
                index = torch.tensor(affine_index, device=x.device)  # [batch_sz, ]
            affine_score = torch.gather(score, dim=-1, index=index.unsqueeze(-1))
            return affine_score
