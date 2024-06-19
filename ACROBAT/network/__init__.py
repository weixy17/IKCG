from . import layer
from . import MaskFlownet
from . import pipeline_img2

def get_pipeline(network, **kwargs):
	if network == 'MaskFlownet':
		return pipeline_img2.PipelineFlownet(**kwargs)
	else:
		raise NotImplementedError
