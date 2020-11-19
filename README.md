### Use main.py as the interface to use the functionality of MLCatchUp.

##Usage Instructions:
- Transformation Inference:  
``
Python main.py --infer <deprecated_api_signature> <updated_api_signature> --output <dsl_script_filepath>
``
- Transformation Application:  
``  
Python main.py -- transform --dsl <dsl_script_filepath> --input <deprecated_filepath> --output <output_filepath>
``
##API Signature Format:  
``
module.function_name(positonal_param1:param_type=param_default_value, *, keyword_param1:param_type=param_default_value)
``

## Examples:  
- Transformation Inference:  
``
Python main.py --infer "torch.gels(input: Tensor, A: Tensor, out=None)" "torch.lstsq(input: Tensor, A: Tensor, out=None)" --output torch_gels.DSL
``

- Transformation Application:  
``
Python main.py --transform --dsl torch_gels.dsl --input input_file.py --output updated_file.py
``