from . import layer
from . import MaskFlownet
from . import pipeline_warp
from . import pipeline_numpy
from . import pipeline_img2
# from . import pipeline

def get_pipeline(network, **kwargs):
	if network == 'MaskFlownet':
		# return pipeline.PipelineFlownet(**kwargs)
		# return pipeline_warp.PipelineFlownet(**kwargs)
		return pipeline_img2.PipelineFlownet(**kwargs)
		# return pipeline_numpy.PipelineFlownet(**kwargs)
	else:
		raise NotImplementedError
