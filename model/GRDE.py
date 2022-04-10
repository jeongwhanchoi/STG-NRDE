import torch
import torch.nn.functional as F
import torch.nn as nn

import torchcde
from vector_fields import *

class NeuralGRDE(nn.Module):
    def __init__(self, args, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol, rtol, solver):
        super(NeuralGRDE, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.logsig_emb_dim = 2

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        
        self.args = args
        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        
        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.emb_opt = False

        self.init_type = 'fc'
        if self.init_type == 'fc':
            if self.emb_opt == False:
                self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
                self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
                # self.initial_z = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            elif self.emb_opt == True:
                self.initial_h = torch.nn.Linear(self.logsig_emb_dim, self.hidden_dim)
                self.initial_z = torch.nn.Linear(self.logsig_emb_dim, self.hidden_dim)        
                self.red_emb = nn.Linear(self.input_dim, self.logsig_emb_dim)
                # self.red_emb = nn.Linear(self.input_dim, self.input_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
            self.start_conv_z = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
        
        self.interpolation = args.spline

        self.adp_opt = False
        if self.adp_opt == True:
            self.proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, times, coeffs):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        # times = torch.linspace(0, len(times)-1, coeffs.size(-2)).to(coeffs.device)
        # times = torch.linspace(0, coeffs.size(-1), coeffs.size(-2)).to(coeffs.device)
        # times = torch.linspace(0, coeffs.size(-2)-1, coeffs.size(-2)).to(coeffs.device)
        # import pdb; pdb.set_trace()
        if self.emb_opt == True:
            coeffs = self.red_emb(coeffs)
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        X0 = X.evaluate(X.interval[0])

        if self.init_type == 'fc':
            h0 = self.initial_h(X0)
            z0 = self.initial_z(X0)
            # z0 = self.initial_z(h0)
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(X0.transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
            z0 = self.start_conv_z(X0.transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()

        
        if self.args.model_type == 'rde':
            z_T = torchcde.cdeint(X=X,
                                func=self.func_g,
                                z0=z0,
                                t=times,
                                adjoint=True,
                                method=self.solver
                                )
        elif self.args.model_type == 'rde2':
            step_size = (X.grid_points[1:] - X.grid_points[:-1]).min()
            # adjoint_params = tuple(self.func_f.parameters()) + tuple(self.func_g.parameters()) + (coeffs,)
            
            z_T = torchcde.cdeint_custom(X=X,
                                func_f=self.func_f,
                                func_g=self.func_g,
                                h0=h0,
                                z0=z0,
                                t=times,
                                adjoint=True,
                                method=self.solver,
                                )

        if self.adp_opt == False:
            z_T = z_T[:,:,-1:,:].transpose(1,2)
        else:
            z_T = z_T.transpose(1,2)
            retain_score = self.proj(z_T)
            retain_score = retain_score.squeeze()
            retain_score = torch.sigmoid(retain_score.transpose(-1,-2))
            retain_score = retain_score.unsqueeze(-1)
            z_T = torch.matmul(retain_score.transpose(-1,-2), z_T.permute(0,2,1,3)).transpose(1,2)

        #CNN based predictor
        # output = self.end_conv(z_T.unsqueeze(-1).shape)                         #B, T*C, N, 1
        output = self.end_conv(z_T)                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output