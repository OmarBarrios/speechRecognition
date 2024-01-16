"""Freezes and optimize the model. Use after training."""
import argparse
import torch
from model import SpeechRecognition
from collections import OrderedDict

def trace(model):
    """
    Traces a PyTorch model using the given input and returns the traced model.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be traced.

    Returns:
        torch.jit.ScriptModule: The traced model.
    """
    model.eval()
    x = torch.rand(1, 81, 300)
    hidden = model._init_hidden(1)
    traced = torch.jit.trace(model, (x, hidden))
    return traced

def main(args):
    """
    A function that loads a saved model checkpoint, creates a new instance of a SpeechRecognition model, 
    loads the state dict of the saved model checkpoint, removes the "model." prefix from the keys of the 
    state dict, loads the new state dict into the model, traces the model, and saves the traced model to 
    a specified save path.

    Args:
        args (Namespace): An object that contains the command-line arguments.

    Returns:
        None
    """
    print("loading model from", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    h_params = SpeechRecognition.hyper_parameters
    model = SpeechRecognition(**h_params)

    model_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    print("tracing model...")
    traced_model = trace(model)
    print("saving to", args.save_path)
    traced_model.save(args.save_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruebas del motor de detección de palabra de activación (wakeword).")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True,
                        help='Punto de control del modelo para optimizar.')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='Ruta para guardar el modelo optimizado.')

    args = parser.parse_args()
    main(args)
