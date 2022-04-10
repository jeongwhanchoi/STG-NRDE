from vector_fields import *
from GFRDE import *
from GFRDE_spatial import *
from GRDE import *
from GCDE import *
from GCDE_temporal import *
from GCDE_spatial import *
from GCDE_Prime import *
from GCDE_Type_I_zprime import *

def make_model(args):
    if args.model_type == 'fft':
        vector_field_f = FinalTanh_f_fft(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_f2 = FinalTanh_f_fft(input_channels=args.input_dim*2, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_g = VectorField_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        vector_field_g2 = VectorField_g(input_channels=args.input_dim*2, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        # vector_field_g = VectorField_g_fft(input_channels=args.input_dim, hidden_channels=args.hid_dim,
        #                                 hidden_hidden_channels=args.hid_hid_dim,
        #                                 num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
        #                                 g_type=args.g_type)
        model = NeuralGFRDE(args, func_f=vector_field_f, func_f2=vector_field_f2, func_g=vector_field_g, func_g2=vector_field_g2, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_g
    elif args.model_type == 'fft_spatial':
        vector_field_g = VectorField_g_fft_spatial(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        model = NeuralGFRDE_spatial(args, func_f=None, func_g=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_g
    elif args.model_type == 'rde':
        vector_field_f = FinalTanh_f_rde(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_g = VectorField_only_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        model = NeuralGRDE(args, func_f=vector_field_f, func_g=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_g
    elif args.model_type == 'rde2':
        vector_field_f = FinalTanh_f_rde(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_g = VectorField_g_rde(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        # vector_field_f = FinalTanh_f_rde(input_channels=2, hidden_channels=args.hid_dim,
        #                                 hidden_hidden_channels=args.hid_hid_dim,
        #                                 num_hidden_layers=args.num_layers)
        # vector_field_g = VectorField_g_rde(input_channels=2, hidden_channels=args.hid_dim,
        #                                 hidden_hidden_channels=args.hid_hid_dim,
        #                                 num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
        #                                 g_type=args.g_type)
        model = NeuralGRDE(args, func_f=vector_field_f, func_g=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_g
    elif args.model_type == 'type1':
        vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_g = VectorField_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        model = NeuralGCDE(args, func_f=vector_field_f, func_g=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_g
    elif args.model_type == 'type1_temporal':
        vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        model = NeuralGCDE_temporal(args, func_f=vector_field_f, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f
    elif args.model_type == 'type1_spatial':
        vector_field_g = VectorField_only_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        model = NeuralGCDE_spatial(args, func_g=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_g
    #TODO:
    elif args.model_type == 'type2': #residual
        vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_f_prime = FinalTanh_f_prime(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_g = VectorField_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        model = NeuralGCDE_PRIME(args, func_f=vector_field_f, func_f_prime=vector_field_f_prime, func_g=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_f_prime, vector_field_g
        
    elif args.model_type == 'type2gg': #residual
        vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_g_prime = VectorField_g_prime(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        vector_field_g = VectorField_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        model = NeuralGCDE_PRIME(args, func_f=vector_field_f, func_f_prime=vector_field_g_prime, func_g=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_g_prime, vector_field_g

    elif args.model_type == 'type1_zprime': #gff'+gf
        vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_g = VectorField_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        vector_field_g_prime = VectorField_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        model = NeuralGCDE_TWICE(args, func_f=vector_field_f, func_g=vector_field_g, func_g_prime=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_g, vector_field_g_prime