import torch
from gpytorch import kernels

class RBFWithLinearTransformationKernel(kernels.Kernel):
    def __init__(self, input_dim, latent_dim, **kwargs):
        """
        Kernel combining Radial Basis Function (RBF) and Linear transformations.

        This kernel applies a linear transformation to the input features before
        computing the RBF kernel on the transformed inputs.

        Parameters
        ----------
        input_dim : int
            Number of input dimensions.
        latent_dim : int
            Number of dimensions in the latent space.
        **kwargs : keyword arguments
            Additional keyword arguments passed to the base kernel constructor.
        """
        super(RBFWithLinearTransformationKernel, self).__init__(**kwargs)
        self.rbf_kernel = kernels.RBFKernel(ard_num_dims=input_dim)
        self.linear_kernel = kernels.LinearKernel(input_dim=input_dim, output_dim=latent_dim)
        self.register_parameter(name="linear_weight", parameter=torch.nn.Parameter(torch.randn(latent_dim, input_dim)))

    def forward(self, x1, x2, **params):
        """
        Calculate the combined RBF kernel value between two sets of inputs.

        Parameters
        ----------
        x1 : torch.Tensor
            First set of input data.
        x2 : torch.Tensor
            Second set of input data.
        **params : keyword arguments
            Additional keyword arguments to be passed to the RBF kernel.

        Returns
        -------
        torch.Tensor
            Combined RBF kernel value between x1 and x2.
        """
        # Apply the linear transformation to the input features
        x1_transformed = torch.matmul(self.linear_kernel(x1), self.linear_weight)
        x2_transformed = torch.matmul(self.linear_kernel(x2), self.linear_weight)
       
        # Calculate the RBF kernel on the transformed inputs
        rbf_kernel_value = self.rbf_kernel(x1_transformed, x2_transformed, **params)
        return rbf_kernel_value
