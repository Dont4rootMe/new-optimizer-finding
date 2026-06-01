## CORE_GENES
### STATE_REPRESENTATION
- Store one compact state accumulator per parameter tensor.

### GRADIENT_PROCESSING
- Normalize raw gradients by a running magnitude estimate.

### UPDATE_RULE
- Apply a damped first-order update using the processed gradient.

### PARAMETER_GROUP_POLICY
- Treat all trainable tensors uniformly unless their gradient norm is zero.

### STEP_CONTROL_POLICY
- Use a short warmup followed by gradual decay.

### STABILITY_POLICY
- Clip unusually large processed updates before application.

### PARAMETERS
- Use conservative default coefficients for decay and clipping.

### OPTIONAL_CODE_SKETCH
- None.

## INTERACTION_NOTES
The state, gradient processing, and update rule form one optimizer.

## COMPUTE_NOTES
The optimizer uses low-overhead tensor operations.

## CHANGE_DESCRIPTION
This optimizer tests compact normalized first-order updates with conservative stability controls.
